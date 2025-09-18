import matplotlib.pyplot as plt
import numpy as np

def extract_lift_drag_data(log_file):
    """Extract lift and drag data from a log file."""
    with open(log_file, 'r') as f:
        lines = f.readlines()

    time_drag = []
    dragx = []
    dragy = []

    for _, l in enumerate(lines):
        if 'dragx' in l:
            time_drag.append(float(l.split()[1]))
            dragx.append(float(l.split()[4]))
        if 'dragy' in l:
            dragy.append(float(l.split()[4]))

    return time_drag, dragx, dragy

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


def compare_lift_drag(file1, file2):
    """Compare lift and drag time history of two files."""
    # Extract data from both files
    time1, drag1, lift1 = extract_lift_drag_data(file1)
    time2, drag2, lift2 = extract_lift_drag_data(file2)
    
    # Create figure with two subplots for drag and lift
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot drag comparison
    ax1.plot(time1, drag1, 'b-', label=f'Drag - {file1}')
    ax1.plot(time2, drag2, 'r--', label=f'Drag - {file2}')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Drag')
    ax1.set_title('Drag Comparison')
    ax1.legend()
    ax1.grid(True, linestyle='--', linewidth=0.5)
    
    # Plot lift comparison
    ax2.plot(time1, lift1, 'b-', label=f'Lift - {file1}')
    ax2.plot(time2, lift2, 'r--', label=f'Lift - {file2}')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Lift')
    ax2.set_title('Lift Comparison')
    ax2.legend()
    ax2.grid(True, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()
    
    # Print average values
    print(f'File 1 ({file1}): Mean Lift: {np.average(lift1)} Mean Drag: {np.average(drag1)}')
    print(f'File 2 ({file2}): Mean Lift: {np.average(lift2)} Mean Drag: {np.average(drag2)}')


if __name__ == "__main__":
    mode = input("Choose mode (1: Single file, 2: Compare two files): ")
    
    if mode == "1":
        log_file_name = input('Input log file name: ')
        plot_lift_drag_strouhal_from_nek_log(log_file_name)
    elif mode == "2":
        log_file_1 = input('Input first log file name: ')
        log_file_2 = input('Input second log file name: ')
        compare_lift_drag(log_file_1, log_file_2)
    else:
        print("Invalid mode selected.")