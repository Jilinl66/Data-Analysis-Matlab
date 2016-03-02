working_directory = '~/Documents/Kaggle/'
train_file = paste0(working_directory,'data2/train.csv')
test_file = paste0(working_directory,'data2/test.csv')
d.train = read.csv(train_file)
str(d.train)
head(d.train)
