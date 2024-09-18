import matplotlib.pyplot as plt

def plot_accuracy(accuracy_list, title='Model Accuracy'):
    plt.figure()
    plt.plot(accuracy_list)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()
