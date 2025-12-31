
import torch
import torch.optim as optim
import numpy as np
import time
import os

import pinn_config as config
import data
import model
import physics
import matplotlib.pyplot as plt

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
    optimizer_adam = optim.Adam(pinn.parameters(), lr=config.LEARNING_RATE)
    
    # Learning rate scheduler: reduce by 0.3 every epochs_adam//5 steps
    scheduler = optim.lr_scheduler.StepLR(optimizer_adam, step_size=config.EPOCHS_ADAM//5, gamma=0.3)
    
    # Data Container
    training_data = data.get_data()
    
    # History
    loss_history = []
    
    print("Starting Adam Training...")
    start_time = time.time()
    last_time = start_time
    
    for epoch in range(config.EPOCHS_ADAM):
        optimizer_adam.zero_grad()
        
        # Periodic data refresh (optional, computationally expensive to re-sample every Step)
        if epoch % 500 == 0 and epoch > 0:
            training_data = data.get_data()
            
        loss_val, losses = physics.compute_loss(pinn, training_data, device)
        loss_val.backward()
        optimizer_adam.step()
        scheduler.step()  # Update learning rate
        
        loss_history.append(loss_val.item())
        
        if epoch % 100 == 0:
            current_time = time.time()
            step_duration = current_time - last_time
            last_time = current_time
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch}: Total Loss: {loss_val.item():.6f} | "
                  f"PDE: {losses['pde']:.6f} | BC_sides: {losses['bc_sides']:.6f} | "
                  f"Free_top: {losses['free_top']:.6f} | Free_bot: {losses['free_bot']:.6f} | "
                  f"Load: {losses['load']:.6f} | LR: {current_lr:.2e} | Time: {step_duration:.4f}s")
            
    print(f"Adam Training Complete. Total Time: {time.time() - start_time:.2f}s")
    
    # L-BFGS Training
    print("Starting L-BFGS Training...")
    optimizer_lbfgs = optim.LBFGS(pinn.parameters(), 
                                  max_iter=1000, 
                                  history_size=50, 
                                  line_search_fn="strong_wolfe")
        
    num_lbfgs_steps = config.EPOCHS_LBFGS
    print(f"Running {num_lbfgs_steps} L-BFGS outer steps.")
    
    for i in range(num_lbfgs_steps):
        # Resample collocation points for each L-BFGS step
        training_data = data.get_data()
        
        def closure():
            optimizer_lbfgs.zero_grad()
            loss_val, _ = physics.compute_loss(pinn, training_data, device)
            loss_val.backward()
            return loss_val
        
        step_start = time.time()
        loss_val = optimizer_lbfgs.step(closure)
        step_end = time.time()
        loss_history.append(loss_val.item())
        
        # Print every step to see progress
        print(f"L-BFGS Step {i}: Loss: {loss_val.item():.6f} | Time: {step_end - step_start:.4f}s")
            
    # Save Model
    torch.save(pinn.state_dict(), "pinn_model.pth")
    np.save("loss_history.npy", np.array(loss_history))
    print("Model saved.")
    return pinn

if __name__ == "__main__":
    train()
