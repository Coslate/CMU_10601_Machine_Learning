import os
if os.getenv("DISPLAY") is None:
    print("No display server available. Switching to 'Agg' backend.")
    import matplotlib
    matplotlib.use('Agg')