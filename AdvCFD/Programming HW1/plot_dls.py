import matplotlib.pyplot as plt
import numpy as np

def plot_lift_drag_strouhal_from_nek_log(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()

    time_drag, time_str = [], []
    dragx, dragy, strouhal = [], [], []

    for _, l in enumerate(lines):
        if 'dragx' in l:
            time_drag.append(float(l.split()[1]))
            dragx.append(float(l.split()[4]))
        if 'dragy' in l:
            dragy.append(float(l.split()[4]))
        if 'Strouhal' in l:
            time_str.append(float(l.split()[1]))
            strouhal.append(float(l.split()[2]))

    print(f'Mean Lift: {np.average(dragy)} Mean Drag: {np.average(dragx)} Mean Strouhal: {np.average(strouhal)}')

    # Plotting dragx and dragy over time
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time_drag, dragx, label='Drag')
    plt.plot(time_drag, dragy, label='Lift')
    plt.xlabel('Time')
    plt.ylabel('Force')
    plt.title('Drag and Lift over Time')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)

    # Scatter plot for Strouhal over time
    plt.subplot(2, 1, 2)
    plt.scatter(time_str, strouhal, label='Strouhal', color='r')
    plt.xlabel('Time')
    plt.ylabel('Strouhal')
    plt.title('Strouhal over Time')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    log_file_name = input('input log file name: ')
    plot_lift_drag_strouhal_from_nek_log(log_file_name)