import sys
import torch
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy


def main():
    path = sys.argv[1]
    results = torch.load(path)
    
    # Train
    train_acc = torch.FloatTensor(results['tracker']['train_acc'])
    train_acc = train_acc.mean(dim=1).numpy()
    train_error = numpy.subtract(numpy.ones(len(train_acc)), train_acc)   
    train_loss = torch.FloatTensor(results['tracker']['train_loss'])
    train_loss = train_loss.mean(dim=1).numpy()    
      
    # Val
    val_acc = torch.FloatTensor(results['tracker']['val_acc'])
    val_acc = val_acc.mean(dim=1).numpy()
    val_error = numpy.subtract(numpy.ones(len(val_acc)), val_acc)
    val_loss = torch.FloatTensor(results['tracker']['val_loss'])
    val_loss = val_loss.mean(dim=1).numpy()
    
    # Plot Error and Loss graphs
    epochs = range(1,51)
    
    plt.figure()
    plt.plot(epochs, train_error, label='Train')
    plt.plot(epochs, val_error, label = 'Valid')
    plt.legend()
    plt.suptitle("Train & Validation Error vs. Epochs") 
    plt.title('Train Error: ' + str(round((1-train_acc[-1])*100, 2)) + '; Validation Error: ' + str(round((1-val_acc[-1])*100, 2)))
    plt.savefig('Error.png')
    
    plt.figure()
    plt.plot(epochs, train_loss, label='Train')
    plt.plot(epochs, val_loss, label = 'Valid')
    plt.legend()
    plt.suptitle("Train & Validation Loss vs. Epochs") 
    plt.title('Train Loss: ' + str(numpy.round(float(train_loss[-1]), 3)) + '; Validation Loss: ' + str(numpy.round(float(val_loss[-1]), 3)))
    plt.savefig('Loss.png')


if __name__ == '__main__':
    main()
