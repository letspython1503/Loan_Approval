#ECO2406, ECO2407, ECO2416
#Source: Kaggle
#################### IMPORTING LIBRARIES AND DATASETS #################################################################################################

### Import libraries.
library(tidyverse)
library(dplyr)
library(caTools)
library(ggplot2)
library(rlang)
library(pROC)
library(gridExtra)
library(corrplot)
library(GGally)
library(rpart)
library(randomForest)
library(caret)

### Load Data.
df <- read.csv("data/loan_approval.csv")
str(df)  # Display information about the Data Frame before cleaning.
glimpse(df)

#################### DATA CLEANING ########################################################################################################################

### Cleaning the Data.
colSums(is.na(df)) # View the structure... Display NA values.
df <- df %>% drop_na() #Removing NA values.
#Left trimming all the spaces on the left of each categorical variable.
for (i in colnames(df)) {
  if (grepl(" ", df[1, i])) {
    df[[i]]<-trimws(df[[i]], which = "left")
  }
}
#Converting from string to numerical.
for (i in colnames(df)) {
  # Check if the first element contains a comma
  if (grepl(",", as.character(df[1, i]))) {
    # Convert to character, trim spaces, remove commas, convert to numeric
    df[[i]]<-as.numeric(gsub(",", "", trimws(as.character(df[[i]]))))
  }
}
#Dropping na once more.
colSums(is.na(df)) # View the structure... Display NA values.
df<-df %>% drop_na() #Removing NA values.
# Display information about the Data Frame after cleaning.
str(df)

#################### Checking the data's Basic statistics #################################################################################################

### Display data and summary
print(colnames(df))
print(head(df))
print(tail(df))
str(df)
### Display summary about the Data Frame.
summary(df)

################### EXPLORATORY DATA ANALYSIS ############################################################################################################

###Split data frame into train and test data
split<-sample.split(df$loan_status, SplitRatio = 0.75)
train_data<-subset(df, split == TRUE)
test_data<-subset(df, split == FALSE)
train_data2<-train_data
test_data2<-test_data

###T-tests for All Numerical Variables vs Loan Status
train_data$loan_status <- as.factor(train_data$loan_status)
for (i in colnames(train_data)){
  if (is.numeric(train_data[[i]])){
    cat("T-test for:", i, "\n")
    formula<-as.formula(paste(i, "~ loan_status"))
    print(t.test(formula, data = train_data))
    cat("\n------------------------\n")
  }
}

###To perform anova tests to discover insights about how the loan has been approved for the numerical variable columns.
for (j in colnames(train_data)){
  if (is.numeric(train_data[[j]])){
    cat("Annova test for:", j, "\n")
    print(summary(aov(reformulate("loan_status", response = j), data = train_data)))
    cat("\n------------------------\n")
  }
}

###To perform chi square tests to discover insights about how the loan has been approved for the categorical variable columns.
for (k in colnames(train_data)) {
  if (is.numeric(train_data[[k]])) {
    next
  } else if(k != "loan_status") {
    cat("Chi-Square test for:", k, "\n")
    print(chisq.test(table(train_data[[k]], train_data$loan_status)))
    cat("\n------------------------\n")
  }
}


################### VISUALIZATION #########################################################################################################################
num_data<-train_data[, sapply(train_data, is.numeric)]
num_data<-select(num_data,-loan_id)

###Box_plot and histogram for each numerical variable from test_data and train_Data
train_data$Dataset <- "Train"
test_data$Dataset <- "Test"
full_data<-bind_rows(train_data, test_data)
num_data_v<-full_data %>% select_if(is.numeric) %>% names()

plots <- lapply(num_data_v, function(var) {
  #Box plot
  boxplot<-ggplot(full_data, aes(x = Dataset, y = .data[[var]], fill = Dataset)) +
    geom_boxplot(alpha = 1) +
    theme_minimal() +
    labs(title = paste("Box Plot of", var), y = var, x = "Dataset") +
    scale_fill_manual(values = c("Train" = "#D8A062", "Test" = "#C1BCD3"))
  # Histogram with density
  hist_plot<-ggplot(full_data, aes(x = .data[[var]], fill = Dataset, color = Dataset)) +
    geom_histogram(aes(y = ..density..), alpha = 0.5, bins = 30, position = "identity") +
    geom_density(alpha = 0.1) +
    theme_minimal() +
    labs(title = paste("Histogram of", var, "[Train, Test]"), y = "Density", x = var) +
    scale_fill_manual(values = c("Train" = "#D8A062", "Test" = "#C1BCD3")) +
    scale_color_manual(values = c("Train" = "#D8A062", "Test" = "#C1BCD3"))
  # Arrange plots side by side
  grid.arrange(boxplot, hist_plot, ncol = 2)
})
plots

###Correlation heat map
cor_matrix<-cor(num_data, use = "complete.obs")
p_matrix<-cor.mtest(num_data)$p 
corrplot(cor_matrix,
         method = "color",
         type = "lower",
         p.mat = p_matrix,
         sig.level = 1,
         addCoef.col = "black",
         insig="blank",
         addgrid.col = "black")

#Stacked Bar graph for categorical variables.
for (z in names(train_data2)) {
  if (!is.numeric(train_data[[z]]) & z != "loan_status") {
    p<-ggplot(train_data2, aes_string(x = z, fill = "loan_status")) +
      geom_bar(position = "fill") +
      ylab("Proportion") +
      ggtitle(paste("Loan Status by", z)) +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    print(p)
  }
}

#Loan_Status approved vs Rejected
ggplot(train_data, aes(x=loan_status,fill=loan_status))+
  geom_bar(width=0.5,color="black",show.legend=FALSE)+
  scale_fill_manual(values=c("Approved"="blue","Rejected"="red"))+
  labs(title = "Loan Status: Approved vs. Rejected",x = "Loan Status",y = "Count")+
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.text = element_text(size = 12),
    axis.title = element_text(face = "bold"))

################### MACHINE LEARNING #########################################################################################################################

###Logistical regression

#Converted Loan Status to a factor since logistic regression requires a categorical dependent variable.
cvt_01<-function(dataframe){
  for (i in 1:nrow(dataframe)) {
    if (dataframe$loan_status[i] == "Approved") {
      dataframe$loan_status[i] <- 1
    } else {
      dataframe$loan_status[i] <- 0
    }
  }
  dataframe$loan_status<-as.numeric(dataframe$loan_status)
  return(dataframe)
}
train_data<-cvt_01(train_data2)
test_data<-cvt_01(test_data2)

log_model<-glm(loan_status ~ ., data = train_data, family = binomial)
summary(log_model)

test_data$predicted_prob<-predict(log_model, newdata = test_data, type = "response")
test_data$predicted_class<-ifelse(test_data$predicted_prob > 0.5, 1, 0)
test_data$predicted_class<-as.factor(test_data$predicted_class)

#Confusion Matrix
conf_matrix<-table(Predicted = test_data$predicted_class, Actual = test_data$loan_status)

#Accuracy, Precision and Recall
log_accuracy<-sum(diag(conf_matrix)) / sum(conf_matrix)
log_precision<-conf_matrix[2,2] / sum(conf_matrix[2,])
log_recall<-conf_matrix[2,2] / sum(conf_matrix[,2])
print(paste("Accuracy:", log_accuracy))
print(paste("Precision:", log_precision))
print(paste("Recall:", log_recall))

log_roc<-roc(test_data$loan_status, test_data$predicted_prob)
print(paste("AUC:", auc(log_roc)))

###Decision Tree
train_data$loan_status<-as.factor(train_data$loan_status)
test_data$loan_status<-as.factor(test_data$loan_status)

tree_model<-rpart(loan_status ~ ., data = train_data, method = "class")
tree_pred_prob<-predict(tree_model, newdata = test_data, type = "prob")[,2]
tree_pred_class<-ifelse(tree_pred_prob > 0.5, 1, 0)
tree_pred_class<-as.factor(tree_pred_class)

# Confusion Matrix
conf_mat<-confusionMatrix(tree_pred_class, test_data$loan_status, positive = "1")

#Accuracy, Precision and Recall
dt_accuracy<-conf_mat$overall["Accuracy"]
dt_precision<-conf_mat$byClass["Precision"]
dt_recall<-conf_mat$byClass["Recall"]
print(paste("Accuracy:", dt_accuracy))
print(paste("Precision:", dt_precision))
print(paste("Recall:", dt_recall))

tree_roc<-roc(test_data$loan_status, tree_pred_prob)
tree_auc<-auc(tree_roc)

###Random Forest
rf_model<-randomForest(loan_status ~ ., data = train_data, ntree = 100)
rf_pred_prob<-predict(rf_model, newdata = test_data, type = "prob")[,2]
rf_pred_class<-ifelse(rf_pred_prob > 0.5, 1, 0)
rf_pred_class<-as.factor(rf_pred_class)

# Confusion Matrix
rf_conf_mat<-confusionMatrix(rf_pred_class, test_data$loan_status, positive = "1")

#Accuracy, Precision and Recall
rf_accuracy<-rf_conf_mat$overall["Accuracy"]
rf_precision<-rf_conf_mat$byClass["Precision"]
rf_recall<-rf_conf_mat$byClass["Recall"]
print(paste("Accuracy:", rf_accuracy))
print(paste("Precision:", rf_precision))
print(paste("Recall:", rf_recall))

rf_roc<-roc(test_data$loan_status, rf_pred_prob)
rf_auc<-auc(rf_roc)

#Comparing ROC curves
plot(log_roc, col = "blue", lwd = 2, main = "ROC Curve Comparison")
lines(tree_roc, col = "green", lwd = 2)
lines(rf_roc, col = "red", lwd = 2)
legend("bottomright",
       legend = c(paste("Logistic Tree AUC:", round(log_auc, 3)),
                  paste("Decision Tree AUC:", round(tree_auc, 3)),
                  paste("Random Forest AUC:", round(rf_auc, 3))),
       col = c("blue", "green", "red"), lwd = 2)


###Comparing Accuracies
accuracy_df <- data.frame(
  Model = c("Logistic Regression", "Decision Tree", "Random Forest"),
  Accuracy = c(log_accuracy, dt_accuracy, rf_accuracy)
)
ggplot(accuracy_df, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", width = 0.6) +
  geom_text(aes(label = round(Accuracy, 3)), vjust = -0.5, size = 4) +
  theme_minimal() +
  labs(title = "Model Accuracy Comparison", x = "Model", y = "Accuracy") +
  theme(legend.position = "none")

################### HYPOTHESIS #########################################################################################################################
#HYPOTHESIS 1
print(summary(aov(train_data2$cibil_score~train_data2$loan_status)))
boxplot(cibil_score ~ loan_status,
        data = train_data2,
        col = "gold",
        main = "CIBIL Score vs Loan Status",
        xlab = "Loan Status",
        ylab = "CIBIL Score")
#HYPOTHESIS 2
train_data2$no_of_dependents <- as.factor(train_data2$no_of_dependents)
print(summary(aov(income_annum~no_of_dependents, data = train_data2)))

boxplot(income_annum ~ no_of_dependents, data = train_data2,
        col = "skyblue",
        main = "Annual Income by Number of Dependents",
        xlab = "Number of Dependents",
        ylab = "Annual Income",
        border = "black")
#HYPOTHESIS 3
train_data2$Total_Assets<-train_data2$residential_assets_value+train_data2$commercial_assets_value+train_data2$luxury_assets_value+train_data2$bank_asset_value
boxplot(Total_Assets~loan_status,data=na.omit(train_data2[,c("Total_Assets","loan_status")]),
        main="Total Assets vs Loan Status",
        xlab="Loan Status",
        ylab="Total Assets",
        col=c("lightblue", "lightgreen"))
print(summary(aov(Total_Assets~loan_status,data=na.omit(train_data2[,c("Total_Assets","loan_status")]))))
