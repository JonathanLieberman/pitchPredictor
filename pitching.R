# Title: pitchingPredictor
# Author: Jonathan Lieberman
# Data Source: baseballsavant.mlb.com
# Style: lowerCamelCase


# This project is an experimental project to determine whether pitches can be predicted
# based on the status of the game and the way a pitch is thrown. The data from 
# baseballsavant.mlb.com is Cubs pitching data from the 2018 MLB season.


# Packages
library(tidyverse)
library(caret)
library(vtreat)

# Personal file path
basePath <- file.path("C://Users//david//OneDrive//Documents//Learning//MLB")
fileName <- "cubsPitching.csv"

# Read in the file
setwd(basePath)
cubsPitching <- read_csv(fileName)


# Should strings be read as factors?
stringsAsFactors <- FALSE

# Should pitching zones be read as factors?
zonesAsFactors <- TRUE

# Should player IDs be read as factors?
IDsAsFactors <- TRUE

# Should Feature Engineering be done?
addFeatures <- TRUE


# Read in pitching data
pitching <- readPitchingData(fileName = fileName
                             , stringsAsFactors = stringsAsFactors
                             , zonesAsFactors = zonesAsFactors
                             , IDsAsFactors = IDsAsFactors
                             )


# Removing Depreciated columns
pitching <- cleanColumns(pitching)


# Feature Engineering
if (addFeatures) {
  pitching <- simpleFeatures(pitching)
}


# Declare model variables
target <- "pitch_type"
vars <- c("player_name"
          , "release_pos_x"
          , "release_pos_y"
          , "release_pos_z"
          , "inning"
          , "p_throws"
          , "stand"
          , "same_side"
          , "balls"
          , "strikes"
          , "net_score"
          , "man_on_1b"
          , "man_on_2b"
          , "man_on_3b"
) 


# Drop variables not in dataset (i.e. no Feature Engingeering)
vars <- vars[vars %in% colnames(pitching)]


# Set training controls
ctrl <- trainControl(method = "cv"
                     , number = 3
                     , allowParallel = TRUE
                     , verboseIter = TRUE
                     , search = "random"
                     )


# Create grid for hyperparameter grid/random search
xgbTreeGrid <- expand.grid(nrounds = c(50, 75, 100)
                           , max_depth = c(5, 6)
                           , eta = c(.3, .4)
                           , gamma = c(0, 1)
                           , colsample_bytree = c(.4)
                           , min_child_weight = c(.5, .6)
                           , subsample = c(.75)
                           )


# Build and tune model
output <- buildSimpleModel(data = pitching
                           , variables = vars
                           , target = target
                           , control = ctrl
                           , xgbTreeGrid = xgbTreeGrid
                           , holdout = .25
                           )


# Parse output from training
model <- output$model
fmla <- output$formula
trainData <- output$trainData
testData <- output$testData
plan <- output$plan


# Use preprocess the testing data
testClean <- preProcessPitching(testData
                                , plan = plan
                                , ignore = target
                                )$data


# Predict on the test data
testData <- testData %>% 
  mutate(pred = predict(model, newdata = testClean))


# Confusion matrix
(confustion <- table(testData$pitch_type, testData$pred))


# Model accuracy
(modelAccuracy <- mean(testData$pitch_type == testData$pred))


# Aggregate pitches by count
trainCounts <- trainData %>%
  group_by(pitch_type) %>%
  summarize(count = n()) %>%
  mutate(percent = count/sum(count))


# Baseline = guess most frequent pitch
mostFrequentPitch <- trainCounts %>% 
  filter(count == max(count)) %>% 
  select(pitch_type) %>%
  as.character()
  
# Baseline accuracy
(baselineAccuracy <- mean(testData$pitch_type == mostFrequentPitch))



# NEXT STEPS

# Find a model which can be understood by a batter
# Switch from caret to parsnip?
# Abstract the model to any team?
# Publish?