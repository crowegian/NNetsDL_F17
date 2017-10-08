import numpy as np

## Here is a training function
def train(model,  X_train, y_train, X_valid, y_valid, 
          num_epoch=10, batch_size=500, learning_rate=1e-3, learning_decay=0.95, verbose=False):
    '''
    This function is for training
    
    Inputs:
    - model: a neural netowrk class object
    - X_train: (float32) input data, a tensor with shape (N, D1, D2, ...)
    - y_train: (int) label data for classification, a 1D array of length N
    - X_valid: (float32) input data, a tensor with shape (num_valid, D1, D2, ...)
    - y_valid: (int) label data for classification, a 1D array of length num_valid
    - num_epoch: (int) the number of training epochs
    - batch_size: (int) the size of a single batch for training
    - learning_rate: (float)
    - learning_decay: (float) reduce learning rate every epoch
    - verbose: (boolean) whether report training process
    '''
    num_train = X_train.shape[0]
    num_batch = num_train//batch_size
    print('number of batches for training: {}'.format(num_batch))
    train_acc_hist = []
    val_acc_hist = []
    # randOrder = np.random.choice(num_train, replace = False)
    bestAcc = 0
    prevAcc = 0
    maxIncreaseCount = 2
    valIncreaseCount = 0
    for e in range(num_epoch):
        # Train stage
        randOrder = np.random.permutation(num_train)
        for i in range(num_batch):
            ## Order selection
            selection = randOrder[i*batch_size:(i+1)*batch_size]
            X_batch = X_train[i*batch_size:(i+1)*batch_size]
            y_batch = y_train[i*batch_size:(i+1)*batch_size]
            ## Random selection
            # sample_idxs = np.random.choice(num_train, batch_size)
            X_batch = X_train[selection,:]
            y_batch = y_train[selection]
            ## loss
            loss = model.loss(X_batch, y_batch)
            # update model
            model.step(learning_rate=learning_rate)
            
            if verbose and (i+1)%10==0:
                print('{}/{} loss: {}'.format(batch_size*(i+1), num_train, loss))
            
        # Validation stage
        sample_idxs = np.random.choice(num_train, 1000)
        train_acc = model.check_accuracy(X_train[sample_idxs,:], y_train[sample_idxs])
        train_acc_hist.append(train_acc)
        
        val_acc = model.check_accuracy(X_valid, y_valid)
        val_acc_hist.append(val_acc)
        # Shrink learning_rate
        learning_rate *= learning_decay
        print('epoch {}: valid acc = {}, new learning rate = {}'.format(e+1, val_acc, learning_rate))
        if str(type(model)) != "<class 'ecbm4040.classifiers.mlp.MLP'>":
            if bestAcc > val_acc:
                valIncreaseCount += 1
                print('Validation acc has not improved for {} turns'.format(valIncreaseCount))
            else:
                valIncreaseCount = 0
            if bestAcc < val_acc:
                bestAcc = val_acc
                model.save_model()
                print('\n'*2)
                print('Best accuracy of {} found. Saving model'.format(val_acc))
                print('\n'*2)
            # print('\n\n\n\ncurr increase count {}\n\n\n\n'.format(valIncreaseCount))
            if valIncreaseCount == maxIncreaseCount:
                print('\n\n Early stopping because of validation loss increase')
                return train_acc_hist, val_acc_hist
        prevAcc = val_acc
        
    # Return Loss history
    return train_acc_hist, val_acc_hist
        
## Test 
def test(model, X_test, y_test):
    test_acc = model.check_accuracy(X_test, y_test)
    print('test acc: {}'.format(test_acc))
    return test_acc