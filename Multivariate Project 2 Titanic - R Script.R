# Installing the necessary packages
if(!require("robustX")) install.packages("robustX")
if(!require("dplyr")) install.packages("dplyr")
if(!require("MASS")) install.packages("MASS")

# Loading the necessary packages
require(robustX); library(robustbase); library(dplyr); library(MASS); 
library(nnet); library(ggplot2); library(gridExtra); library(car)

# Loading the data set
df = read.csv("titanic.csv")

# Viewing the first five rows of the data set
head(df, 5)

# Outputting all of the column/variable names
colnames(df)

# Viewing the dimensions of the data set
dataset_dimensions = dim(df)
cat("Number of Observations/Rows:", dataset_dimensions[1], "\n")
cat("Number of Variables/Columns:", dataset_dimensions[2], "\n")

# Selecting numeric variables
numeric_vars = c("Age", "Siblings.Spouses.Aboard", "Parents.Children.Aboard", "Fare")
df_numeric = df[, numeric_vars]
head(df_numeric, 5)

# Checking the Assumption of Normality
# Defining a function to generate qqnorm plots for all 4 quantitative vars. using both the qqnorm and qqline functions
qqnorm_plot_generator = function(variable, quantitative_variable_name) {
  qqnorm(variable, main = paste("Q-Q Plot of", quantitative_variable_name), 
         xlab = "Theoretical Quantiles", ylab = "Observed Values")
  qqline(variable, col = "red")
}

par(mfrow = c(2,2))

qqnorm_plot_generator(df$Age, "Age")
qqnorm_plot_generator(df$Siblings.Spouses.Aboard, "Siblings & Spouses Aboard")
qqnorm_plot_generator(df$Parents.Children.Aboard, "Parents & Children Aboard")
qqnorm_plot_generator(df$Fare, "Fare")

# Applying Internal Validation for FLDA
# ------------------------------ FLDA -------------------------------- #
flda = function(x,class) {
  cat("Fisher Linear Discriminant:\n")
  a = lda(x, class)
  d = predict(a)      
  t = table(class, d$class) 
  print(t)
  er = 100 * (sum(t) - sum(diag(t))) / nrow(x)
  cat("Error Rate =", er, "%\n")
  return(d)
}

# Using the categorical variable with at least 3 classes in it
categorical_var_with_3_classes = df$Pclass

flda_result = flda(df_numeric, categorical_var_with_3_classes)

# Applying Internal Validation for Multinomial Logistic Regression
# ---------------------------- Multinomial ------------------------------ #
mn = multinom(Pclass ~ . - Name - Sex - Survived, data = df)

results = predict(mn)

table(df$Pclass, results)

# Applying External Validation for FLDA
# ------------------------------ FLDA -------------------------------- #
set.seed(123)

# Remove 'Name' before splitting due to it being a factor & the FLDA method cannot handle new factor levels in the test set that were not in the training set
df_clean = df[, !colnames(df) %in% c("Name")]

# Split data into 70% training & 30% testing
train_index = sample(1:nrow(df_clean), 0.7 * nrow(df_clean))
train_data = df_clean[train_index, ]
test_data = df_clean[-train_index, ]

# Train FLDA model on training data
flda_model = lda(Pclass ~ . - Sex - Survived, data = train_data)

# Predict on test data
flda_predictions = predict(flda_model, test_data)$class

# Confusion matrix and error rate for FLDA External Validation
flda_table = table(test_data$Pclass, flda_predictions)
print(flda_table)

cat("\n")

flda_external_error_rate = 100 * (sum(flda_table) - sum(diag(flda_table))) / nrow(test_data)
cat("External Validation Error Rate (FLDA) =", flda_external_error_rate, "%")

# Applying External Validation for Multinomial Logistic Regression
# ------------------------------ Multinomial -------------------------------- #
set.seed(123)

mn_model = multinom(Pclass ~ . - Sex - Survived, data = train_data)

mn_predictions = predict(mn_model, test_data)

mn_table = table(test_data$Pclass, mn_predictions)
print(mn_table)

# Print an empty line for legibility reasons
cat("\n")

mn_external_error_rate = 100 * (sum(mn_table) - sum(diag(mn_table))) / nrow(test_data)
cat("External Validation Error Rate (Multinomial Logistic Regression) =", mn_external_error_rate, "%")

# Apply FLDA2 to 2 of the 3 Groups/Classes within the Categorical Variable
# ------------------------------ FLDA2 -------------------------------- #
# Subset the data for 2 out of the 3 classes within the variable "Pclass"
# Subset which includes the 1st and 3rd-class passengers & leaves out the 2nd-class passengers
df_subset = df[df$Pclass %in% c(1, 3), ]

# Selecting only the relevant quantitative variables
relevant_numeric_vars = df_subset[, c("Fare", "Age")]

# Class Labels (1 & 3)
class_subset = as.factor(df_subset$Pclass)

# Apply FLDA2
lda_model = lda(relevant_numeric_vars, class_subset)

# Projection Method
lda_projection = predict(lda_model, relevant_numeric_vars)

# Plotting the projected values
plot(lda_projection$x, col = class_subset, pch = 19, main = "FLDA2 Plot: First vs Third Class Passengers", 
     xlab = "LD1", ylab = "LD2")  

# Legend to label which class is which
legend("topright", legend = levels(class_subset), col = 1:length(levels(class_subset)), pch = 19)

cat("Confusion Matrix:")
print(table(class_subset, lda_projection$class))

# Print an empty line for legibility reasons
cat("\n")

# Calculate error rate
error_rate = 100 * (sum(table(class_subset, lda_projection$class)) - sum(diag(table(class_subset, lda_projection$class)))) / nrow(df_subset)
cat("FLDA2 Classification Error Rate =", error_rate, "%")


