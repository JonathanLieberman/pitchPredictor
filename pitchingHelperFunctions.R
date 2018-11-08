# Title: pitchingHelperFunctions
# Author: Jonathan Lieberman
# Data Source: baseballsavant.mlb.com
# Style: lowerCamelCase


# This script contains the source code for a handful of helper functions used in the 
# pitching.R script.

# Create function to read pitch data from baseballsavant.mlb.com
readPitchingData <- function(fileName
                               , stringsAsFactors = FALSE
                               , zonesAsFactors = FALSE
                               , IDsAsFactors = FALSE
                             ) {
  
  pitching <- read_csv(fileName
                       , col_types = list(pitch_type = col_character() # type of pitch thrown
                                          , game_date = col_date() # date game was played
                                          , release_pos_x = col_double() # release positions
                                          , release_pos_y = col_double()
                                          , release_pos_z = col_double()
                                          , player_name = col_character() # character name of pitcher
                                          , batter = col_integer() # integer ID of batter
                                          , pitcher = col_integer() # integer ID of pitcher
                                          , events = col_character() # outcome of the pitch (null = not in play)
                                          , description = col_character() # descriptions are discrete and finite (i.e. hit_in_play)
                                          , zone = col_integer() # zones are numbers, but not ordinal
                                          , game_type = col_character() # game types are either regular or post season
                                          , stand = col_character() # batters bat from the right or left
                                          , p_throws = col_character() # handedness of the pitcher
                                          , home_team = col_character()
                                          , away_team = col_character()
                                          , type = col_character() # Ball, strike, or in play
                                          , hit_location = col_character() # baseball positions are not ordinal
                                          , bb_type = col_character() # ground ball, line drive, or fly ball
                                          , pfx_x = col_double()
                                          , pfx_z = col_double()
                                          , plate_x = col_double()
                                          , plate_z = col_double()
                                          , on_3b = col_integer() # player id for runner on 3rd base
                                          , on_2b = col_integer() # player id for runner on 2nd base
                                          , on_1b = col_integer() # player id for runner on 1st base
                                          , inning_topbot = col_character() # top or bottom of inning
                                          , hc_x = col_double()
                                          , hc_y = col_double()
                                          , vx0 = col_double()
                                          , vy0 = col_double()
                                          , vz0 = col_double()
                                          , ax = col_double()
                                          , ay = col_double()
                                          , az = col_double()
                                          , hit_distance_sc = col_double()
                                          , launch_speed = col_double()
                                          , launch_angle = col_double()
                                          , effective_speed = col_double()
                                          , release_spin_rate = col_double()
                                          , release_extension = col_double()
                                          , estimated_ba_using_speedangle = col_double()
                                          , estimated_woba_using_speedangle = col_double()
                                          , woba_value = col_double()
                                          , woba_denom = col_double()
                                          , babip_value = col_double()
                                          , iso_value = col_double()
                                          , launch_speed_angle = col_double()
                                          , if_fielding_alignment = col_character()
                                          , of_fielding_alignment = col_character()
                                          )
                       )
  
  # Create list of variables to turn into factors
  toFactor <- c()
  
  if (stringsAsFactors) {
    strings <- map_lgl(pitching, is_character) %>%
      which %>%
      names
    toFactor <- c(toFactor, strings)
  }
  
  if (zonesAsFactors) {
    toFactor <- c(toFactor, "zone")
  }
  
  if (IDsAsFactors) {
    toFactor <- c(toFactor, "batter", "pitcher")
  }
  
  # Factor columns
  pitching <- pitching %>%
    mutate_at(toFactor, as.factor)
}



# Remove depreciated columns
cleanColumns <- function(df) {
  # remove depreciated columns
  df %>% 
    select(-spin_dir
           , -spin_rate_deprecated
           , -break_angle_deprecated
           , -break_length_deprecated
           , -tfs_deprecated
           , -tfs_zulu_deprecated
           ) %>%
    return()
}


# Feature Engineering
simpleFeatures <- function(df) {
  df %>%
    mutate(same_side = (stand == p_throws) # are the pitcher and batter using the same side?
           , cubs_home = home_team == "CHC" # are the cubs home?
           , man_on_3b = !is.na(on_3b) # boolean variable for if runner is on 3rd
           , man_on_2b = !is.na(on_2b) # boolean variable for if runner is on 2nd
           , man_on_1b = !is.na(on_1b)# boolean variable for if runner is on 1st
           , net_score = (away_score - home_score) * (-1 ^ cubs_home) # net score relative to Cubs
           )
}


# Prepare pitching data for modelling
preProcessPitching <- function(df
                              , plan = NULL # default to creating a new plan
                              , ignore = NULL # default to preprocess all variables
                           ) {
  
  varnames <- colnames(df)
  varnames <- varnames[!(varnames %in% ignore)]
  
  # Create simple treatment plan if missing
  # missing categoricals -> new factor level
  # missing numerics -> imputed mean, NEW is_bad column
  if (is.null(plan)) {
    plan <- designTreatmentsZ(df, varlist = varnames)
  }
  
  
  # Implement plan
  df_clean <- vtreat::prepare(plan, df)
  
  
  if (!is.null(ignore)) {
    # Reattach ignored columns
    df_clean <- cbind(df[, ignore], df_clean)
  }
  
  # Always return list for return type consistency
  return(list(data = df_clean, plan = plan))
}


# Create formula
createFormula <-function(target
                         , indVariables
                         , logy = FALSE
                         , nonPredictiveVars = NULL
                         ) {
    # Create RHS of formula
    indVariables <- indVariables[!(indVariables %in% c(nonPredictiveVars, target))]
    RHS <- paste(indVariables, collapse = " + ")
    
    
    # Create LHS of formula
    if (logy) {
      LHS <- paste0("log(", target, ")")
    } else {
      LHS <- target
    }
    
    
    # Add LHS of formula
    fmla <- as.formula(paste(LHS, RHS, sep = " ~ "))
    return(fmla)
}


# Build a model and test it
buildSimpleModel <- function(data
                             , variables
                             , target
                             , control
                             , xgbTreeGrid
                             , holdout = .25
) {
  
  # Subset data
  dataSubset <- data %>%
    filter(pitch_type != "null") %>% # drop rows with no pitch type
    select(c(target, variables)) %>%
    droplevels() # Not every pitcher throws all pitch types
  
  
  # Partition data
  inTrain <- createDataPartition(y = dataSubset$pitch_type
                                 , p = 1 - holdout
                                 , list = FALSE
  )
  
  trainData <- dataSubset[inTrain,]
  testData <- dataSubset[-inTrain,]
  
  
  # Preprocessing
  pOut <- preProcessPitching(trainData
                             , ignore = target
  )
  trainClean <- pOut$data
  plan <- pOut$plan
  
  
  # Create model formula
  fmla <- createFormula(target = target
                        , indVariables = colnames(trainClean)
                        , logy = FALSE
  )
  
  
  # Train model
  model <- train(fmla
                 , data = trainClean
                 , method = "xgbTree"
                 , metric = "Accuracy"
                 , preProc = c("nzv", "center", "scale")
                 , na.action = na.pass
                 , trControl = ctrl
                 , tuneGrid = xgbTreeGrid
                 , tuneLength = 1
  )
  
  output <- list(model = model
                 , formula = fmla
                 , trainData = trainData
                 , testData = testData
                 , plan = plan
  )
}