#---
#Code for predicting housing prices
#Written by Lovisa Stenström 221101
#---
  
### INSTALL PACKAGES ###
install.packages("dplyr")
install.packages("corrplot")
install.packages("tidyverse")

### LOAD LIBRARIES###
library(dplyr)
library(corrplot)
library(tidyverse)

### READ IN DATA ###
housing_data <- read.csv(file = "housing-data.csv",
                         stringsAsFactors = TRUE)

#Show subset of data
head(housing_data)

### DATA EXPLORATION ###
#First, let's have a quick look at the summary statistics of the dataset.
summary(housing_data)

#We can for example see that some houses have been sold for 0$, we will remove those later.
#We also want to check NA values. 
anyNA(housing_data)

#There is no present in the data.

#We also want to check outliers. 
boxplot(housing_data$price)

#There are some prices here that stand out from the rest, we will probably remove at least the top two later,
#since these might confound the fitted regression line.
#Then, since we will predict housing prices, we want to make sure that there's no remarkable change of price over time in the dataset.
plot(rownames(housing_data),housing_data$price)

#This plot shows that it's not. It is only sold houses during a two month period here, so that makes sense.

### First feature selection ###
#After we quickly checked the quality of our data, we will start looking more closely into the variables.
#We can directly see that one variable isn't neccesary: Country. All houses are sold in the US. Therefore we can remove that feature. 
#Since we also just concluded that time is not affecting the price for this dataset, we will exclude that as well.

#There are also three other features that describe the geographic position of the house in three different ways: street, city and statezip.
#For now, we will proceed only with the city feature, as it is more detailed than state zip code, but less detailed than street.
transformed_data <- select(housing_data, -c(date, country, street, statezip))


### DATA CLEAN-UP ###
#Now when we have the features we will start building our regression from, we need to transform the categorical variables into numerical.
transformed_data$city <- as.numeric(transformed_data$city)
length(unique(transformed_data$city))

#City now ranges from 1-44, which means that there are 44 different cities present in the dataset.
#Now our data frame consists of a mix of numericals and integers, so we will transform our data to only be numerical.
transformed_data <- mutate_all(transformed_data, function(x) as.numeric(x))

#We want to exclude the houses sold for 0$, so we remove those. We will also remove the two biggest outliers.
transformed_data <- transformed_data %>% filter(price != 0)
transformed_data <- transformed_data %>% arrange(desc(price))
transformed_data <- transformed_data[-c(1,2),]
boxplot(transformed_data$price)

#Now the boxplot looks a bit better.

### SECOND FEATURE SELECTION ###

#We don't want to include features that highly correlate, so let's create a correlation matrix.
correlations <- cor(transformed_data)
corrplot(correlations, type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)


#We can see here that sqft living and sqft above are highly correlated, so we will not include sqft above.
transformed_data <- select(transformed_data, -sqft_above)

### TRAIN MODEL ###
#Now we can start training the first model. First, we need to split the data into training and test set.
#We will go for 80% in the training set, and 20% of the data in the test set.
#(Setting seed to get reproducible result)
set.seed(417)
idx <- sample(nrow(transformed_data), nrow(transformed_data)* 0.80)
housing_train <- transformed_data[idx,]
housing_test <- transformed_data[ -idx,]

#And now we will create our first simple linear regression to predict housing prices.
model_1 <- lm(price ~ bedrooms + bathrooms + sqft_living + sqft_lot + floors + waterfront + 
views + condition + sqft_basement + yr_built + yr_renovated + city,
data = housing_train)
summary(model_1)

#The summary of the regression shows us that the R squared is only 0.57, which isn't a really impressive fit.
#Additionally, the year renovated seems to be poorly correlated with house prices.
#We don't even want to procceed with this model, so we will try to engineer the featureas a bit in the next section.

### FEATURE ENGINEERING ###
#Since the original features don't really fit a great line to the data, we will try to engineer the features to better predict housing prices.
#Everyone that bought a housing knows that there are at least two more relevant variables determining the price: how many houses that are available in the area and price/sqft.
#Let's try to add those features based on the data we have.

#Add availability as a metric based on number of houses sold in the area during 2 month (the time period included in the dataset).
transformed_data$availability <- sapply(transformed_data$city,
                                        function(x) length(which(transformed_data$city == x)))

#Let's split the new data into training and test set again, and then train our second model.
idx <- sample(nrow(transformed_data), nrow(transformed_data)* 0.80)
housing_train <- transformed_data[idx,]
housing_test <- transformed_data[ -idx,]

model_2 <- lm(price ~ bedrooms + bathrooms + sqft_living + sqft_lot + floors + waterfront +
views + condition + sqft_basement + yr_built + yr_renovated + city +
availability, data = housing_train)
summary(model_2)

#We can see here that the availability seems to increase the fit a bit, but not much.
#Let's try with adding price/sqft.
transformed_data$price_p_sqft <- transformed_data$price/transformed_data$sqft_living
idx <- sample(nrow(transformed_data), nrow(transformed_data)* 0.80)
housing_train <- transformed_data[idx,]
housing_test <- transformed_data[ -idx,]

model_3 <- lm(price ~ bedrooms + bathrooms + sqft_living + sqft_lot + floors + waterfront +
                views + condition + sqft_basement + yr_built + yr_renovated + city +
                availability + price_p_sqft, data = housing_train)
summary(model_3)

#We can now see that the price/sqft drastically increase the fit to R squared = 0.88, which is much better than previously.
#It is also evident that most other features are not significant anymore when introducing the price/sqft variable.
#So those can probably be dropped without influencing model performance. However, we will move on and evaluate the predictions of our model now.

### EVALUATING MODEL ###
#Let's predict the prices on our test set.
housing_test$pred <- predict(model_3, housing_test)
predictions <- housing_test[,c(1,16)]
predictions$relative_error <- abs(housing_test$price - housing_test$pred)/housing_test$price #This won't translate well to the predictions under 0, but they are not so many.

#We can have a quick look at a few predictions.
predictions[1:20,]

#And then we will make some visualizations of our predictions.
#How is the fit to the regression line?
plot(model_3)

#How is the distribution of price predictions?
hist(housing_test$pred)

#We can directly spot one big flaw here in the predictions, some houses have been predicted to have a negative price value.
#This is obviously not possible but we will go ahead and estimate the error of our model using Root Mean Square Error anyway.
error_3 <- sqrt(mean((housing_test$price - housing_test$pred)^2))
print(error_3)

#This is a fairly large error and there are for sure room for improvement in the model.
#If we try to predict the prices using our second model, is the error better or worse?
housing_test$pred <- predict(model_2, housing_test)
error_2 <- sqrt(mean((housing_test$price - housing_test$pred)^2))
print(error_2)

#The error is much bigger for our second model, meaning that the featues engineering at least improved our model.

#Is there a relationship between real house price and relative error of the prediction?
plot(predictions$price, predictions$relative_error)

#Yes, it seems that the model has larger relative errors for the lower priced houses.

#### FUTURE IMPROVEMENTS ###

#Currently, we have a model that is fairy well fitted to the training data, but it performs quite bad, especially for lower priced houses.
#Maybe lower priced houses have been bought for less than asked price, and therefore the price doesn't follow what it is "supposed" to cost.
#But that is only speculation. Based on this result, it is aslo possible that the housing prices cannot really be predicted using a linear model,
#as prices might behave differently if they are cheap or expensive.

#There is still room for improvement in this model, one big is to get rid of negative predictions. If I had more time on building this model,
#I would probably look into possibilities of using other regression techniques, to ensure that a negative value cannot be predicted and explore non-linear fittings.

#When the model is good enough to not predict negative values, the error could also be computed using the Root Mean Square Logatitmic Error.
#This relative measure is more accurate in this case, as both lower and higher priced houses will contribute equally to the error, compared to measuring absolute error.
#However, it is not possible to compute that on negative values.

#Another thing is to look more into is to try to normalize or scale the features. This is something that is not included in this draft.

#Some features can probably also be removed since they don't contribute much to the predictions

#Otherwise, I would also look more into engineering the existing features to make them more descriptive and better at predicting house prices.
#For example, mean housing prices in each city/area.

#And as a last remark, this model doesn't at all take price changes over time into account, as probably is the case on most markets.
#For that, more advanced algorithms like taking moving averages into account would be needed.
