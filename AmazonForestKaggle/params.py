img_dir = '/content/train-tif-v2'
path_to_csv = '/content/train_v2.csv'
im_size = 224
img_dir = './train-tif-v2'
batch_size = 16
epoch = 10
test_size = 0.2
optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.5)
