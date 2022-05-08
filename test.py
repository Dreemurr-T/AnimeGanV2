import os
name = 'generator'
epoch = 1
save_dir = f"checkpoint/{name}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir,f"{epoch}.pth")
print(save_path)