#########################
# Load Libraries + Data #
#########################
library(DataExplorer)
library(tidymodels)
library(lubridate)
library(tidyverse)
library(recipes)
library(xgboost)
library(glmnet)
library(ranger)
library(embed)
library(lme4)

setwd("C:/Users/HongtingWang/Documents/STAT 348 - Predictive Analytics/KaggleRestaurantRevenuePrediction/data")

train <- read_csv("./train.csv")
test <- read_csv("./test.csv")



#############################
# EDA + Feature Engineering #
#############################

####################################
# Basic Structure + Missing Values #
####################################
glimpse(train)
glimpse(test)

# Check Missing Values
colSums(is.na(train))
colSums(is.na(test))



############################
# Target Variable: Revenue #
############################
# Summary Statistics
summary(train$revenue)

# Distribution + Density
ggplot(train, aes(x = revenue)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "white") +
  labs(title = "Revenue Distribution")

ggplot(train, aes(x = revenue)) +
  geom_density(fill = "lightblue") +
  labs(title = "Revenue Density")

# Remove Outliers
train <- train %>% filter(revenue < 15000000)


##########################################################
# Extract Day, Month, Year, and Days Open from Open Date #
##########################################################
train <- train %>%
  mutate(
    OpenDate = mdy(`Open Date`),
    Day = as.numeric(format(OpenDate, "%d")),
    Month = as.factor(format(OpenDate, "%m")),
    Year = as.factor(format(OpenDate, "%y")),
    DaysOpen = as.numeric(Sys.Date() - OpenDate)
  )

test <- test %>%
  mutate(
    OpenDate = mdy(`Open Date`),
    Day = as.numeric(format(OpenDate, "%d")),
    Month = as.factor(format(OpenDate, "%m")),
    Year = as.factor(format(OpenDate, "%y")),
    DaysOpen = as.numeric(Sys.Date() - OpenDate)
  )

glimpse(train)
glimpse(test)


#########################
# Categorical Variables #
#########################
# Check Category Counts
train %>% 
  count(City) %>% 
  arrange(desc(n)) %>%
  head(20)

length(unique(train$City))

train %>% count(`City Group`)

train %>% count(Type)

# Convert Low-Cardinality Categories to Factors
train$CityGroup <- factor(train$`City Group`)
train$Type <- factor(train$Type)

test$CityGroup <- factor(test$`City Group`)
test$Type <- factor(test$Type)

# Remove Non-predictive and Redundant Fields
train_data <- train %>%
  select(-c(Id, City, `City Group`, `Open Date`, OpenDate))

test_data <- test %>%
  select(-c(Id, City, `City Group`, `Open Date`, OpenDate))

glimpse(train_data)
glimpse(test_data)

# Bar Plots
ggplot(train_data, aes(x = CityGroup)) +
  geom_bar(fill = "steelblue")

ggplot(train_data, aes(x = Type)) +
  geom_bar(fill = "darkgreen")



##################################
# Obfuscated Features (P1 - P37) #
##################################
# Correlation with Revenue
p_cols <- paste0("P", 1:37)

cor_data <- train_data %>%
  select(all_of(p_cols), revenue) %>%
  cor()

# Heatmap of Correlations
DataExplorer::plot_correlation(train_data %>%
                                 select(all_of(p_cols), revenue))

# Identify Top Correlated Features
cor_values <- cor_data[, "revenue"] %>%
  sort(decreasing = TRUE)

head(cor_values, 11)



##########################################
# Train vs. Test Distribution Comparison #
##########################################
compare_cols <- c("P2", "P28", "P17", "P6", "DaysOpen")

plot_density(train_data[, compare_cols])
plot_density(test_data[, compare_cols])



######################
# Define the Recipes #
######################
# Recipe for Linear Regression
recipe_linear <- recipe(revenue ~ ., data = train_data) %>%
  
  # Handle Unseen Levels in Test Set
  step_unknown(all_nominal_predictors()) %>%
  step_novel(all_nominal_predictors()) %>%

  # Dummy Encode Low-Cardinality Categorical Predictors
  step_dummy(all_nominal_predictors()) %>%
  
  # Remove Zero-Variance Columns
  step_zv(all_predictors()) %>%
  
  # Scale Numeric Features
  step_normalize(all_numeric_predictors(), -all_outcomes())

# Recipe for Tree Models
recipe_trees <- recipe(revenue ~ ., data = train_data) %>%
  
  # Handle Unseen Levels in Test Set
  step_unknown(all_nominal_predictors()) %>%
  step_novel(all_nominal_predictors()) %>%

  # Dummy Encode Low-Cardinality Categorical Predictors
  step_dummy(all_nominal_predictors()) %>%
  
  # Remove Zero-Variance Columns
  step_zv(all_predictors())



############
# Modeling #
############

##########################
# Elastic Net Regression #
##########################
# Model Specification
el_model <- linear_reg(
  penalty = tune(),
  mixture = 0.6
) %>%
  set_engine("glmnet")

# Model Workflow
el_wf <- workflow() %>%
  add_recipe(recipe_linear) %>%
  add_model(el_model)

# Hyperparameter Grid
el_grid <- grid_regular(
  penalty(range = c(-5, 10)),
  levels = 20
)

# Cross-Validation Tuning
set.seed(123)

el_res <- tune_grid(
  el_wf,
  resamples = vfold_cv(train_data, v = 5),
  grid = el_grid,
  metrics = metric_set(rmse)
)

show_best(el_res, metric = "rmse", n = 3)

best_el <- select_best(el_res, metric = "rmse")
best_el

# Final Fit on the Train Set
final_el <- finalize_workflow(el_wf, best_el) %>%
  fit(train_data)

# Generate Predictions on the Test Set
el_preds <- predict(final_el, test_data)

# Format Submission File
el_submission <- test %>%
  select(Id) %>%
  bind_cols(el_preds) %>%
  rename(Prediction = .pred)

write_csv(el_submission, "el_submission.csv")


#################
# Random Forest #
#################
# Model Specification
rf_model <- rand_forest(
  mtry = tune(),
  trees = 35,
  min_n = tune(),
) %>%
  set_engine("ranger") %>%
  set_mode("regression")

# Model Workflow
rf_wf <- workflow() %>%
  add_recipe(recipe_trees) %>%
  add_model(rf_model)

# Hyperparameter Grid
set.seed(123)

rf_grid <- grid_random(
  mtry(range = c(10, 20)),
  min_n(range = c(5, 20)),
  size = 35
)

# Cross-Validation Tuning
set.seed(123)

rf_res <- tune_grid(
  rf_wf, 
  resamples = vfold_cv(train_data, v = 5),
  grid = rf_grid,
  metrics = metric_set(rmse)
)

show_best(rf_res, metric = "rmse", n = 10)

best_rf <- select_best(rf_res, metric = "rmse")
best_rf

# Final Fit on the Train Set
final_rf <- finalize_workflow(rf_wf, best_rf) %>%
  fit(train_data)

# Generate Predictions on the Test Set
rf_preds <- predict(final_rf, test_data)

# Format Submission File
rf_submission <- test %>%
  select(Id) %>%
  bind_cols(rf_preds) %>%
  rename(Prediction = .pred)

write_csv(rf_submission, "rf_submission.csv")



###########
# XGBoost #
###########
# Model Specification
xgb_model <- boost_tree(
  trees = tune(),
  learn_rate = tune(),
  mtry = tune(),
  min_n = tune(),
  tree_depth = tune(),
  loss_reduction = tune(),
  sample_size = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("regression")

# Model Workflow
xgb_wf <- workflow() %>%
  add_recipe(recipe_trees) %>%
  add_model(xgb_model)

# Hyperparameter Grid
set.seed(123)

xgb_grid <- grid_random(
  trees(range = c(20, 35)),
  learn_rate(range = c(0.1, 0.5), trans = NULL),
  mtry(range = c(10, 20)),
  min_n(range = c(5, 20)),
  tree_depth(range = c(1, 3)),
  loss_reduction(range = c(0, 5), trans = NULL),
  sample_size = sample_prop(range = c(0.6, 1.0)),
  size = 50
)

# Cross-Validation Tuning
set.seed(123)

xgb_res = tune_grid(
  xgb_wf,
  resamples = vfold_cv(train_data, v = 5),
  grid = xgb_grid,
  metrics = metric_set(rmse)
)

show_best(xgb_res, metric = "rmse", n = 10)

best_xgb = select_best(xgb_res, metric = "rmse")
best_xgb

# Final Fit on the Train Set
final_xgb <- finalize_workflow(xgb_wf, best_xgb) %>%
  fit(train_data)

# Generate Predictions on the Test Set
xgb_preds <- predict(final_xgb, test_data)

# Format Submission File
xgb_submission <- test %>%
  select(Id) %>%
  bind_cols(xgb_preds) %>%
  rename(Prediction = .pred)

write_csv(xgb_submission, "xgb_submission.csv")