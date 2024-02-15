class Trainer():
  # loss_function like CrossEntropyLoss
  def __init__(self, model, lr, weight_decay=0.9e-6):
    self.model = model
    self.loss_function = nn.BCELoss()
    self.lr = lr
    self.weight_decay = weight_decay
    self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    self.fold_index_dict = {}

    self.K_fold_flag = False

    self.fold_name = 'None'
    self.report_dataframe = pd.DataFrame(data=[], columns=['Epoch', 'Time', 'Acc', 'Loss', 'ValAcc', 'ValLoss', 'Weights saved', 'FoldName', 'FileName'])

  def LossFunction(self, prediction, target):
    return self.loss_function(prediction, target)

  def Eval(self, X, Y, batch_size=32):
    self.model.eval()

    loss_list = []
    accuracy_list = []
    # forward
    index = np.arange(X.size(0))
    for b in range(int(np.ceil(X.size(0)/batch_size))):
      if b+1 != int(np.ceil(X.size(0)/batch_size)):
        batch_index = index[b*batch_size:(b+1)*batch_size]
      else:
        batch_index = index[b*batch_size:]
        if len(batch_index) == 1:
          break

      y_softmax = self.model(X[batch_index])
      loss = self.LossFunction(y_softmax, Y[batch_index])
      loss_list.append(loss.item())

      accuracy = torch.sum(torch.argmax(Y[batch_index], dim=-1)==torch.argmax(y_softmax, dim=-1), dim=-1)/float(y_softmax.shape[0])
      accuracy_list.append(accuracy)

    return sum(loss_list)/float(len(loss_list)), (sum(accuracy_list)/float(len(accuracy_list)))*100.0

  def Train(self, X, Y, epochs, batch_size,
            x_valid, y_valid,
            save_weight_flag=True, auto_save_weight_acc=None, weight_save_path='', weight_file_name='',
            Seed=None, **kwargs):

    # Set random seed
    if torch.cuda.is_available():
      torch.cuda.manual_seed(Seed)
    else:
      torch.manual_seed(Seed)

    len_data = X.size(0)

    if save_weight_flag:
      assert auto_save_weight_acc != None, 'Please set \"auto_save_weight_acc\" variable'

    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []
    batch_loss_list = []
    batch_acc_list = []
    for epoch in range(epochs):
      self.model.train()
      start_time = time()

      train_index = np.arange(len_data)
      train_index = np.random.permutation(train_index)
      train_index_batch = np.array_split(train_index, indices_or_sections=len_data//batch_size)

      for i, batch in enumerate(train_index_batch):
        #x_batch, y_batch = X[batch,], Y[batch,]

        # forward
        y_softmax = self.model(X[batch,])
        y_train = torch.squeeze(Y[batch,])

        loss = self.LossFunction(y_softmax, y_train)
        accuracy = torch.sum(torch.argmax(y_train, dim=-1)==torch.argmax(y_softmax, dim=-1), dim=0)/float(y_train.size(0))

        # backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        batch_loss_list.append(loss.item())
        batch_acc_list.append(accuracy.item()*100)

        # Disp
        print(f'\r Epoch:{epoch+1} ---- Batch:{i+1}/{len_data//batch_size}, Loss:{loss.item()}',end='')

      # Train loss and accuracy
      train_loss, train_acc = self.Eval(X, Y, batch_size=16)
      train_loss_list.append(train_loss)
      train_acc_list.append(train_acc.item())

      # Valid loss and accuracy
      valid_loss, valid_acc = self.Eval(x_valid, y_valid, batch_size=batch_size)
      valid_loss_list.append(valid_loss)
      valid_acc_list.append(valid_acc.item())

      save_flag = False
      weight_name = '-'
      # Auto save weights
      if save_weight_flag:
        if valid_acc >= auto_save_weight_acc:
          os.makedirs(weight_save_path,exist_ok=True)

          #date = datetime.datetime.now()
          #new_file_name = file_name + '_'
          #date =  date.strftime("%Y_%b_%d_%H_%M_%S")
          #new_file_name += '_' + 'Train_Loss_' + str(train_loss) + '_Valid_Acc_' + str(valid_acc.item())
          new_file_name = '{:s}_LR_{:.3e}_Wdecay_{:.3e}_BatchSize_{:d}_Epoch_{:d}_TrainLoss_{:.3f}_ValidAcc_{:.4f}'.format(weight_file_name, self.lr, self.weight_decay, batch_size, epoch+1, train_loss, valid_acc.item())
          #print(new_file_name)
          weight_name = os.path.join('weights', weight_save_path, new_file_name)
          torch.save(self.model.state_dict(),weight_name)

          if 'Test' in kwargs.keys():
            test_loss, test_acc = self.Eval(X=kwargs['Test'][0], Y=kwargs['Test'][1], batch_size=16)

          save_flag = True

      # print report
      last_report = '\r Epoch:{:03d}-Time:{:04d}s ---- Acc:{:.03f}%, Loss:{:.03f}, ValAcc:{:.03f}%, ValLoss:{:.03f}'.format(epoch+1, int(time()-start_time), train_acc_list[-1], train_loss_list[-1], valid_acc_list[-1], valid_loss_list[-1])
      if save_flag:
        if 'Test' in kwargs.keys():
          last_report += ', Weights saved, Test Acc: {:.03f}%'.format(test_acc)
        else:
          last_report += ', Weights saved, Test Acc: --'
      print(last_report)
      # save report in dataframe
      self.report_dataframe.loc[len(self.report_dataframe.index)] = [epoch+1,
                                                                     int(time()-start_time),
                                                                     train_acc_list[-1],
                                                                     train_loss_list[-1],
                                                                     valid_acc_list[-1],
                                                                     valid_loss_list[-1],
                                                                     save_flag,
                                                                     self.fold_name,
                                                                     weight_name]

    return {'Train_loss':train_loss_list, 'Train_acc':train_acc_list,
            'Valid_loss':valid_loss_list, 'Valid_acc':valid_acc_list,
            'Batch_Train_loss':batch_loss_list, 'Batch_Train_acc':batch_acc_list,
            }

  def TrainKFold(self, X, Y, Valid_split_size:float,
                 epochs:int, batch_size:int,
                 K=None,
                 save_weight_flag=True, auto_save_weight_acc=None, save_path='', file_name='',
                 Seed=None):

    assert K != None, "Please config \"K\""
    # Save first initialize
    copy_model = copy.deepcopy(self.model)

    # Make folder with a specific name
    date = datetime.datetime.now()
    folder_name = str(K) + '_Fold_Training_' + file_name + '_' + date.strftime("%Y_%b_%d_%H_%M_%S")
    folder_path = os.path.join(save_path, folder_name)
    os.makedirs(folder_path,exist_ok=True)

    # Make fold data
    self.MakeDataFold(len_data=X.shape[0], K=K, Seed=Seed, print_flag=True)

    # Train
    train_K_fold_result = {}
    for n_K, self.fold_name in enumerate(self.fold_index_dict.keys()):
      print("***** {:s} Started ***************************************************************************".format(self.fold_name))
      start_time = time()

      # Reset model
      self.model = copy.deepcopy(copy_model)
      self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

      train_index, test_index = self.fold_index_dict[self.fold_name]
      # train_size = int(len(train_index)*(1-Valid_split_size))
      # train_index, valid_index = train_index[:train_size], train_index[train_size:]

      train_K_fold_result[self.fold_name] = self.Train(X=X[train_index], Y=Y[train_index], x_valid=X[test_index], y_valid=Y[test_index],
                                                      epochs=epochs,
                                                      batch_size=batch_size,
                                                      save_weight_flag=save_weight_flag,
                                                      auto_save_weight_acc=auto_save_weight_acc,
                                                      weight_save_path=folder_path,
                                                      weight_file_name=file_name + '_' + self.fold_name,
                                                      Seed=Seed)


      #file.close()
      print("***** {:s} Finished - Time = {:.3f}s *********************************************************".format(self.fold_name, time()-start_time))

    # Save Reports
    with open(os.path.join(folder_path, 'Reports.csv'), 'w') as f:
      self.report_dataframe.to_csv(f)

    with open(os.path.join(folder_path, 'result_dict_pickle'), 'wb') as f:
      pickle.dump(train_K_fold_result, f)

    with open(os.path.join(folder_path, 'index_dict_pickle'), 'wb') as f:
      pickle.dump({'Folds':self.fold_index_dict, 'Valid_split_size':Valid_split_size}, f)

    return train_K_fold_result


  def MakeDataFold(self, len_data, K, Seed=None, print_flag=True):
    self.K_fold_flag = True

    if Seed != None:
      np.random.seed(Seed)

    data_index = np.arange(len_data)

    data_index = np.random.permutation(data_index)
    index_fold = np.array_split(data_index, K)

    #for i in range(len(index_fold)):
    #  print('Length Fold {:d} : {:d}'.format(i+1, len(index_fold[i])))

    for n, fold in enumerate(index_fold):
      fold_name = 'Fold_{:d}'.format(n)

      train_fold = list(range(K))
      train_fold.remove(n)
      test_fold = list(range(K))[n]

      train_index = []
      for f in train_fold:
        train_index.extend(index_fold[f])

      self.fold_index_dict[fold_name] = (train_index, index_fold[test_fold])

    if print_flag:
      print('*************************************************************')
      for f in self.fold_index_dict:
        print('Length Fold \'{:s}\' : Train: {:d}, Test: {:d}'.format(f, len(self.fold_index_dict[f][0]), len(self.fold_index_dict[f][1])))
      print('*************************************************************')

def predicion(model, X):
  model.eval()

  y_pred = []
  for x in X:
    y_pred.append(model(torch.unsqueeze(x, 0)))

  y_pred = torch.stack(y_pred, dim=0)
  y_pred = torch.squeeze(y_pred)
  return y_pred, torch.argmax(y_pred, -1)

def Accuracy(model, X, Y):
  model.eval()

  y_pred = []
  for x in X:
    y_pred.append(model(torch.unsqueeze(x, 0)))

  y_pred = torch.stack(y_pred, dim=0)
  y_pred = torch.squeeze(y_pred)

  acc = torch.sum(torch.argmax(y_pred, -1) == torch.argmax(Y, -1)) / Y.size(0)
  return acc.item()


