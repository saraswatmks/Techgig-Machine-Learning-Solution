path <- "/home/manish/Desktop/Data2017/June/ML/"
setwd(path)


#load data and libraries
library(data.table)
library(bit64)
library(lubridate)

aum <- fread("Code-Gladiators-AUM.csv")
transactions <- fread("Code-Gladiators-Transaction.csv")
investmentexperience <- fread("Code-Gladiators-InvestmentExperience.csv")
activity <- fread("Code-Gladiators-Activity.csv")
bulkemail <- fread("Code-Gladiators-Bulk-Email.csv")
marketreturns <- fread("Code-Gladiators-Market-Returns.csv")

train <- copy(transactions)
rm(transactions)

train[,Month := gsub(pattern = "\\ / ", replacement = "-", x = Month)]
train[,Month := paste0(Month,"-01")]
train[,Month := as.POSIXct(Month, format = "%Y-%m-%d")]
train[,summary(Month)]

investmentexperience[,Month := gsub(pattern = "\\ / ", replacement = "-", x = Month)]
investmentexperience[,Month := paste0(Month,"-01")]
investmentexperience[,Month := as.POSIXct(Month, format = "%Y-%m-%d")]
investmentexperience[,summary(Month)]

aum[,Month := gsub(pattern = "\\ / ", replacement = "-", x = Month)]
aum[,Month := paste0(Month,"-01")]
aum[,Month := as.POSIXct(Month, format = "%Y-%m-%d")]
aum[,summary(Month)]

activity[,Month := gsub(pattern = "\\ / ", replacement = "-", x = Month)]
activity[,Month := paste0(Month,"-01")]
activity[,Month := as.POSIXct(Month, format = "%Y-%m-%d")]
activity[,summary(Month)]

c_activity <- copy(activity)
c_activity[,Activity_Count := .N, .(Unique_Advisor_Id,Month)]
c_activity[,Activity_Type_New := list(list(Activity_Type)), .(Unique_Advisor_Id, Month)]
c_activity[,Activity_Type := NULL]
c_activity <- unique(c_activity,by=c('Unique_Advisor_Id','Month'))

bulkemail[,Year := substr(x = YearMonth, start = 1, stop = 4)]
bulkemail[,mon := substr(x = YearMonth, start = 5, stop = 6)]
bulkemail[,Month := paste(Year,mon,"01",sep = "-")]
bulkemail[,c('Year','mon') := NULL]
bulkemail[,YearMonth := NULL]
bulkemail[,Month := as.POSIXct(Month, format = "%Y-%m-%d")]

head(marketreturns)
marketreturns[,Year := substr(x = month, start = 1, stop = 4)]
marketreturns[,mon := substr(x = month, start = 5, stop = 6)]
marketreturns[,Month := paste(Year,mon,"01",sep = "-")]
marketreturns[,Month := as.POSIXct(Month, format = "%Y-%m-%d")]  
marketreturns[,month := NULL]
marketreturns[,c('Year','mon') := NULL]


# Merge Files -------------------------------------------------------------

# investment experience
train <- investmentexperience[train, on=c('Unique_Investment_Id','Month')]

# asset under management (AUM)
train <- aum[train, on=c('Unique_Investment_Id','Unique_Advisor_Id','Month')]

# activity
train <- c_activity[train, on=c('Unique_Advisor_Id','Month')]

# bulkemail
train <- bulkemail[train, on = c('Unique_Advisor_Id','Month')]

## market returns
train <- marketreturns[train, on ='Month']



# Remove columns with missing values (>80) --------------------------------

dx <- sapply(train, function(x) sum(is.na(x))/length(x))
dx <- dx[dx > 0.8]
dx

# remove variables
train <- train[,-names(dx),with=F]



# Impute missing values in the list ---------------------------------------

library(purrr)

train[,Activity_Type_New := ifelse(map(Activity_Type_New, is_empty),0,Activity_Type_New)]
train[,no_activity := unlist(lapply(Activity_Type_New, function(x) length(x)))]

train[,Activity_Type_New := NULL]
train[,Activity_Count := NULL]


# Encode variables --------------------------------------------------------

train[,`Morningstar Category` := as.integer(as.factor(`Morningstar Category`))]
train[,Investment := as.integer(as.factor(Investment))]

train[,.N/nrow(train),Transaction_Type]
train[,Transaction_Type := ifelse(Transaction_Type == 'R',1,0)]



# Random Guess ------------------------------------------------------------

guss1 <- train[,mean(Transaction_Type),.(Unique_Advisor_Id)]
setnames(guss1, "V1", "mean_pred")

guss1 <- unique(guss1)
head(guss1)

guss2 <- guss1[,mean(mean_pred), Unique_Advisor_Id]
setnames(guss2, "V1", "mean_pred")

test <- fread("test_data.csv")
test <- guss2[test, on=c('Unique_Advisor_Id')]

test[,Redeem_Status := ifelse(mean_pred >= 0.3, "YES","NO")]
setnames(test, "mean_pred","Propensity_Score")

subm2 <- test[,c(1,3,2,4)]
subm2[is.na(Propensity_Score), Propensity_Score := 0]
subm2[is.na(Redeem_Status), Redeem_Status := "NO"]
colSums(is.na(subm2))
fwrite(subm2, "sub_one_base.csv") # 70.69



# Machine Learning Approach -----------------------------------------------

# few unique values
sapply(train, function(x) length(unique(x)))
dontuse <- c('Unique_Investment_Id'
             ,'Unique_Advisor_Id'
             ,'Transaction_Type'
             ,'Month'
             ,'DowJones.return'
             ,'Russell2000.return'
             ,'Nasdaq.return'
             ,'SP500.return')

X_test <- train[Month == '2016-12-01'] #Decemeber
X_train <- train[Month >= "2016-08-01" & Month < "2016-12-01"] #August to November

X_train[,Amount := round(as.numeric(Amount))]
X_test[,Amount := round(as.numeric(Amount))]

dtrain <- X_train[Month <= "2016-10-01"]
dval <- X_train[Month >= "2016-11-01"]

# some weak variables

X_train[,mean_amt_bothid := round(mean(Amount)),.(Unique_Advisor_Id,Unique_Investment_Id)]
X_test[,mean_amt_bothid := round(mean(Amount)),.(Unique_Advisor_Id,Unique_Investment_Id)]

X_train[,mean_amt_advid := round(mean(Amount)),.(Unique_Advisor_Id)]
X_test[,mean_amt_advid := round(mean(Amount)),.(Unique_Advisor_Id)]

X_train[,mean_amt_invid := round(mean(Amount)),.(Unique_Investment_Id)]
X_test[,mean_amt_invid := round(mean(Amount)),.(Unique_Investment_Id)]


# Model Training

d_train <- xgb.DMatrix(data = as.matrix(X_train[,-dontuse, with=F]), label=X_train$Transaction_Type)
d_test <- xgb.DMatrix(data = as.matrix(X_test[,-dontuse, with=F]), label = X_test$Transaction_Type)

params <- list(booster = "gbtree"
               ,objective = "binary:logistic"
               ,eta=0.3
               ,max_depth=6
               ,min_child_weight=1
               ,subsample=0.7
               ,colsample_bytree=0.7
               ,metric = 'error')

set.seed(1)

bst <- xgb.train(params = params
                 ,data = d_train
                 ,nrounds = 500
                 ,watchlist = list(train=d_train, val = d_test)
                 ,print_every_n = 10
                 ,early_stopping_rounds = 20
                 ,maximize = F) 

pred <- predict(bst, d_test)


# Make Submission File

## submission

sub_1 <- data.table(Unique_Advisor_Id = X_test$Unique_Advisor_Id, Pred = pred)
sub_1[,Pred := mean(Pred), Unique_Advisor_Id]

test <- fread("test_data.csv")

sub_1 <- unique(sub_1)

test <- sub_1[test, on='Unique_Advisor_Id']
setnames(test, "Pred", "Propensity_Score")

test <- test[,c(1,3,2)]
test[is.na(Propensity_Score), Propensity_Score := 0]

a4 <- copy(test)
a4[,Redeem_Status := ifelse(Propensity_Score >= 0.7,'YES','NO')]

a5 <- copy(test)
a5[,Redeem_Status := ifelse(Propensity_Score >= 0.15,'YES','NO')]

fwrite(a4,"xgb_two_a4.csv") #submitted - 49.48
fwrite(a5,"xgb_two_a5.csv") #submitted - 51.86


### Ensemble
high <- fread("sub_one_base.csv")
mid <- fread("xgb_two_a5.csv")
low <- fread("xgb_two_a4.csv")

all_redeem <- data.table(high = high$Redeem_Status, mid = mid$Redeem_Status, low = low$Redeem_Status)

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

all_redeem$final <- apply(all_redeem,1,Mode)

head(all_redeem)

final_sub <- copy(high)
final_sub[,Redeem_Status := all_redeem$final]

fwrite(final_sub,"ensemble_preds_2.csv") #68.5656

###################### End ########################################################















































