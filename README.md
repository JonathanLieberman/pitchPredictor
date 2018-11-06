# pitchPredictor
Predicting pitch types based on MLB game status and pitch release

This project was aimed at determining how well an MLB pitch type (fastball, curveball, etc.) can be predicted based on information available right when the ball leaves the pitcher's hand.

Currently, this project uses `caret` (R package) to interface with `xgboost` (C code) to show that predicting a pitch type before the ball starts travelling is possible. The next challenge is developing a model which you can teach a batter.

# Requirements to Run
1. R
2. `tidyverse`, `caret`, and `vtreat` R packages installed
3. Access to `baseballsavant.mlb.com` (free) to pull data

# Quick Start
1. Grab data from `baseballsavant.mlb.com` and save it as a `.csv`
2. Change `basePath` and `fileName` to your local paths
3. Source `pitchingHelperFunctions.R`
4. Run `pitching.R`

# Other information
This project was developed for learning purposes using open source technology. Feel free to take this code and adapt it how you see fit!

For any questions or suggestions, submit an issue or email me at `JonathanLieberman2017@u.northwestern.edu`
