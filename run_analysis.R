########## TASK 1
##--------download the datasets and prepare it to be explored and read by R--------##

library(dplyr) 
#-----test files---------------#
test_subjects <- read.table(file = file.path('UCI HAR Dataset','test','subject_test.txt'))

test_values <- read.table(file = file.path('UCI HAR Dataset','test','X_test.txt'))

test_activity <- read.table(file = file.path('UCI HAR Dataset','test','y_test.txt'))

#--------train files---------#

train_subjects <- read.table(file = file.path('UCI HAR Dataset','train','subject_train.txt'))

train_values <- read.table(file = file.path('UCI HAR Dataset','train','X_train.txt'))

train_activity <- read.table(file = file.path('UCI HAR Dataset','train','y_train.txt'))

#-----features and activity--------#
features <- read.table(file.path('UCI HAR Dataset','features.txt'),as.is = T)

activity <- read.table(file.path('UCI HAR Dataset','activity_labels.txt'))
colnames(activity) <- c('ID','label')

#------merging---------#

acdata <- rbind(
                  cbind(train_subjects,train_values,train_activity),
                  cbind(test_subjects,test_values,test_activity))


colnames(acdata) <- c('subject',features[,2],'activity')

# deleting used single imported dataframes in order free up memory space by those datasets
rm(train_activity,train_subjects,train_values,test_activity,test_subjects,test_values)

########## TASK 2
#-------keeping means stds and subjects or activities---------#

wantedcolumnsvec <- grep(colnames(acdata),pattern = 'subject|activity|mean|std')

acdata <- acdata[,wantedcolumnsvec]

############ TASK 3
#----- Uses descriptive activity names to name the activities in the data set ----#

acdata$activity <- factor ( x = acdata$activity , levels = activity$ID, labels = activity$label)


######## TASK 4
#-----Appropriately labels the data set with descriptive variable names.----#

varname <- colnames(acdata)
varname <- gsub(pattern = '[//()_]-','',varname)
varname <- gsub(pattern = '^t','time-domain',varname)
varname <- gsub(pattern = '^f','frequency-domain ',varname)
varname <- gsub(pattern = 'Acc','accelerometer',varname)
varname <- gsub(pattern = 'Gyro',"gyroscope",varname)
varname <- gsub(pattern = 'Mag','magnitude',varname)
varname <- gsub(pattern = 'std','standarddeviation',varname)

colnames(acdata) <- varname

######## TASK 5
#-----creates a second, independent tidy data set with the average of each variable for each activity and each subject.----#
tidyhumanact <- as_tibble(acdata)
tidyhumanact <- tidyhumanact[,c(1,81,2:80)]

tidyhumanact <- tidyhumanact %>%  group_by(subject,activity) %>%  summarise_all(funs(mean))

write.table(tidyhumanact,'tidy_data.csv', row.names = F,quote = F)
