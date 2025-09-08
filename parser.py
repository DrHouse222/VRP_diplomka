import vrplib

# Read VRPLIB formatted instances (default)
instance = vrplib.read_instance("Set_A/A-n32-k5.vrp")

print(instance['name'])