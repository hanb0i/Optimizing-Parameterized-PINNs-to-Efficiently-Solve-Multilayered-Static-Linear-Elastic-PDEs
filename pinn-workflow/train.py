
import torch
import torch.optim as optim
import numpy as np
import time

from scipy.linalg import cholesky, LinAlgError
from scipy.optimize import minimize
from torch.nn.utils import parameters_to_vector, vector_to_parameters

import soap
import scipy_patch

import pinn_config as config
import data
import model
import physics

def train():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Initialize Model
    pinn = model.MultiLayerPINN().to(device)
    print(pinn)
    
    # Initialize Optimizers
    optimizer_soap = soap.SOAP(
        pinn.parameters(),
        lr=config.LEARNING_RATE,
        betas=(0.95, 0.95),
        weight_decay=0.0,
        precondition_frequency=config.SOAP_PRECONDITION_FREQUENCY,
    )
    
    # Learning rate scheduler: reduce by 0.3 every epochs_soap//5 steps
    scheduler = optim.lr_scheduler.StepLR(optimizer_soap, step_size=config.EPOCHS_SOAP//5, gamma=0.3)
    
    # Load FEM data for comparison
    print("Loading FEM solution for comparison...")
    try:
        fem_data = np.load("fea_solution.npy", allow_pickle=True).item()
        X_fea = fem_data['x']
        Y_fea = fem_data['y']
        Z_fea = fem_data['z']
        U_fea = fem_data['u']
        
        # Prepare FEM evaluation grid
        pts_fea = np.stack([X_fea.ravel(), Y_fea.ravel(), Z_fea.ravel()], axis=1)
        pts_fea_tensor = torch.tensor(pts_fea, dtype=torch.float32).to(device)
        u_fea_flat = U_fea.reshape(-1, 3)
        
        fem_available = True
        print(f"FEM data loaded: {X_fea.shape}")
    except FileNotFoundError:
        print("FEM solution not found. Training without FEM comparison.")
        fem_available = False
    
    # Data Container
    training_data = data.get_data()
    
    # Load Hybrid Training Data (Sparse FEA samples)
    x_fea_train, u_fea_train = data.get_fea_samples(config.N_DATA)
    if len(x_fea_train) > 0:
        print(f"Loaded {len(x_fea_train)} sparse FEA points for hybrid training.")
    
    # Initialize Dataset and DataLoader
    dataset = data.PINNDataset(training_data, fea_data=(x_fea_train, u_fea_train))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    # History - store all loss components separately for each optimizer.
    soap_history = {
        'total': [], 'pde': [], 'bc_sides': [], 'free_top': [], 'free_bot': [], 'load': [], 'data': [],
        'fem_mae': [], 'fem_max_err': [], 'epochs': []
    }
    
    ssbfgs_history = {
        'total': [], 'pde': [], 'bc_sides': [], 'free_top': [], 'free_bot': [], 'load': [], 'data': [],
        'fem_mae': [], 'fem_max_err': [], 'steps': []
    }
    
    print("Starting SOAP Pretraining (Mini-batch Mode)...")
    start_time = time.time()
    last_time = start_time
    
    for epoch in range(config.EPOCHS_SOAP):
        
        # Periodic data refresh with residual-based adaptive sampling
        if epoch % 500 == 0 and epoch > 0:
            residuals = physics.compute_residuals(pinn, training_data, device)
            training_data = data.get_data(prev_data=training_data, residuals=residuals)
            
            # Re-initialize dataset/loader with new samples (keep same FEA data)
            dataset = data.PINNDataset(training_data, fea_data=(x_fea_train, u_fea_train))
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
            print(f"  Resampled with residual-based adaptive sampling at epoch {epoch}")
            
        # --- Mini-batch Loop ---
        epoch_loss = 0.0
        epoch_losses = {}
        n_batches = 0
        
        for batch in dataloader:
            optimizer_soap.zero_grad()
            
            # Transfer batch to device inside compute_loss (it expects dict of tensors now)
            # Actually physics.compute_loss assumes dict of *lists* for 'interior' etc. if coming from get_data
            # BUT our dataset returns tensors directly. We need to adapt physics.py or ensure batch works.
            # physics.py: x_int = data['interior'][0] -> This implies it expects a list!
            # Our PINNDataset returns tensors.
            # Workaround: Wrap tensors in list to make physics.py happy, OR modify physics.py
            # Modifying physics.py to handle both is cleaner, but for now let's wrap here.
            
            # Wait, `physics.compute_loss` does: `x_int = data['interior'][0].to(device)`.
            # If `data['interior']` is a tensor (batch), `[0]` gets the first element! That's bad.
            # I MUST check physics.py again.
            
            # Let's fix physics.py in the next step if strictly needed, but better to fix here if possible?
            # No, `dataset` returns a batch of tensors.
            # If I pass `{'interior': [batch_interior], ...}` it works.
            
            batch_data = {}
            for k, v in batch.items():
                if k in ['interior', 'sides']:
                    batch_data[k] = [v] # Wrap in list to match physics.py expectation
                else:
                    batch_data[k] = v
            
            loss_val, losses = physics.compute_loss(pinn, batch_data, device)
            loss_val.backward()
            optimizer_soap.step()
            
            epoch_loss += loss_val.item()
            for k, v in losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = 0.0
                epoch_losses[k] += v.item()
            n_batches += 1
            
        scheduler.step()  # Update learning rate per epoch
        
        # Average losses for history
        avg_loss = epoch_loss / n_batches
        soap_history['total'].append(avg_loss)
        for k in epoch_losses:
            if k in soap_history:
                soap_history[k].append(epoch_losses[k] / n_batches)
        if 'data' not in epoch_losses: # If data loss wasn't computed (e.g. empty)
             if 'data' in soap_history: soap_history['data'].append(0.0)

        
        if epoch % 100 == 0:
            current_time = time.time()
            step_duration = current_time - last_time
            last_time = current_time
            current_lr = scheduler.get_last_lr()[0]
            
            avg_pde = epoch_losses['pde'] / n_batches
            avg_load = epoch_losses['load'] / n_batches
            avg_data = epoch_losses.get('data', 0.0) / n_batches
            
            # Compute FEM error every 100 epochs
            if fem_available:
                with torch.no_grad():
                    u_pinn_flat = pinn(pts_fea_tensor, 0).cpu().numpy()
                    diff = np.abs(u_pinn_flat - u_fea_flat)
                    mae = np.mean(diff)
                    soap_history['fem_mae'].append(mae)
                    soap_history['fem_max_err'].append(np.max(diff))
                    soap_history['epochs'].append(epoch)
                    
                print(f"Epoch {epoch}: Loss: {avg_loss:.6f} | "
                      f"PDE: {avg_pde:.6f} | Data: {avg_data:.6f} | LR: {current_lr:.2e} | "
                      f"FEM MAE: {mae:.6f} | Time: {step_duration:.4f}s")
            else:
                print(f"Epoch {epoch}: Loss: {avg_loss:.6f} | "
                      f"PDE: {avg_pde:.6f} | Data: {avg_data:.6f} | "
                      f"LR: {current_lr:.2e} | Time: {step_duration:.4f}s")
            
    print(f"SOAP Pretraining Complete. Total Time: {time.time() - start_time:.2f}s")
    
    # SciPy self-scaled BFGS fine-tuning
    print(f"Starting SciPy SSBFGS Fine-Tuning ({config.SS_BFGS_VARIANT})...")
    if scipy_patch.ensure_scipy_bfgs_patch():
        print("Applied local SciPy optimize patch for method_bfgs support.")

    param_device = next(pinn.parameters()).device
    param_dtype = next(pinn.parameters()).dtype

    def _set_params(flat_params):
        flat_tensor = torch.as_tensor(flat_params, dtype=param_dtype, device=param_device)
        with torch.no_grad():
            vector_to_parameters(flat_tensor, pinn.parameters())

    def loss_and_grad(flat_params):
        _set_params(flat_params)
        loss_val, _ = physics.compute_loss(pinn, training_data, device)
        grads = torch.autograd.grad(loss_val, pinn.parameters(), create_graph=False, retain_graph=False)
        grad_flat = torch.cat([g.reshape(-1) for g in grads])
        return float(loss_val.item()), grad_flat.detach().cpu().numpy().astype(np.float64, copy=False)

    num_ssbfgs_steps = config.EPOCHS_SSBFGS
    print(f"Running {num_ssbfgs_steps} SSBFGS outer steps.")
    print("Resampling with residual-based adaptive sampling each outer step.")

    initial_weights = parameters_to_vector(pinn.parameters()).detach().cpu().numpy().astype(np.float64, copy=False)
    hess_inv0 = np.eye(initial_weights.size, dtype=np.float64)

    for i in range(num_ssbfgs_steps):
        # Resample collocation points with residual-based adaptive sampling
        residuals = physics.compute_residuals(pinn, training_data, device)
        training_data = data.get_data(prev_data=training_data, residuals=residuals)

        step_start = time.time()
        result = minimize(
            loss_and_grad,
            initial_weights,
            method=config.SS_BFGS_METHOD,
            jac=True,
            options={
                'maxiter': config.SS_BFGS_MAXITER,
                'gtol': config.SS_BFGS_GTOL,
                'hess_inv0': hess_inv0,
                'method_bfgs': config.SS_BFGS_VARIANT,
                'initial_scale': config.SS_BFGS_INITIAL_SCALE,
            },
            tol=0.0,
        )
        step_end = time.time()

        if not result.success:
            print(f"  SciPy minimize status {result.status}: {result.message}")

        initial_weights = result.x
        _set_params(initial_weights)

        hess_inv0 = getattr(result, "hess_inv", None)
        if isinstance(hess_inv0, np.ndarray):
            hess_inv0 = 0.5 * (hess_inv0 + hess_inv0.T)
            try:
                cholesky(hess_inv0)
            except LinAlgError:
                hess_inv0 = np.eye(len(initial_weights), dtype=np.float64)
        else:
            hess_inv0 = np.eye(len(initial_weights), dtype=np.float64)

        # Compute losses for logging
        loss_val, losses = physics.compute_loss(pinn, training_data, device)
        ssbfgs_history['total'].append(loss_val.item())
        ssbfgs_history['pde'].append(losses['pde'].item())
        ssbfgs_history['bc_sides'].append(losses['bc_sides'].item())
        ssbfgs_history['free_top'].append(losses['free_top'].item())
        ssbfgs_history['free_bot'].append(losses['free_bot'].item())
        ssbfgs_history['load'].append(losses['load'].item())

        # Compute FEM error and print
        if fem_available:
            with torch.no_grad():
                u_pinn_flat = pinn(pts_fea_tensor, 0).cpu().numpy()
                diff = np.abs(u_pinn_flat - u_fea_flat)
                mae = np.mean(diff)
                max_err = np.max(diff)
                ssbfgs_history['fem_mae'].append(mae)
                ssbfgs_history['fem_max_err'].append(max_err)
                ssbfgs_history['steps'].append(i)
            print(f"SSBFGS Step {i}: Total Loss: {loss_val.item():.6e} | PDE: {losses['pde'].item():.6e} | "
                  f"BC_sides: {losses['bc_sides'].item():.6e} | Free_top: {losses['free_top'].item():.6e} | "
                  f"Free_bot: {losses['free_bot'].item():.6e} | Load: {losses['load'].item():.6e} | "
                  f"FEM MAE: {mae:.6e} | Time: {step_end - step_start:.4f}s")
        else:
            print(f"SSBFGS Step {i}: Total Loss: {loss_val.item():.6e} | PDE: {losses['pde'].item():.6e} | "
                  f"BC_sides: {losses['bc_sides'].item():.6e} | Free_top: {losses['free_top'].item():.6e} | "
                  f"Free_bot: {losses['free_bot'].item():.6e} | Load: {losses['load'].item():.6e} | "
                  f"Time: {step_end - step_start:.4f}s")

        # Save model at every SSBFGS step
        torch.save(pinn.state_dict(), "pinn_model.pth")
            
    # Save Model and Loss Histories
    torch.save(pinn.state_dict(), "pinn_model.pth")
    loss_history = {'soap': soap_history, 'ssbfgs': ssbfgs_history}
    np.save("loss_history.npy", loss_history)
    print("Model saved.")
    return pinn

if __name__ == "__main__":
    train()
