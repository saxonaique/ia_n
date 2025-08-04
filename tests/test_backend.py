import matplotlib
print(f"Matplotlib backend: {matplotlib.get_backend()}")
print(f"Backend configuration file: {matplotlib.matplotlib_fname()}")
print(f"Available backends: {matplotlib.rcsetup.interactive_bk}")

# Try to import TkAgg
matplotlib.use('TkAgg', force=True)
from matplotlib import pyplot as plt
print("Successfully imported pyplot with TkAgg backend")

# Try to create a simple figure
fig = plt.figure()
print("Successfully created figure")
plt.close(fig)
print("Test completed successfully")
