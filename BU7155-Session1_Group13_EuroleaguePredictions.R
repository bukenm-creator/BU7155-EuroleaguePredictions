library(dplyr)
library(lubridate)
library(RcppRoll) 
library(tidyr)    
library(scales)   
library(lme4)     

print("--- Libraries Loaded ---")

tryCatch({
  header <- read.csv("euroleague_header.csv")
  box_score <- read.csv("euroleague_box_score.csv")
  print("--- Step 1: Data Loaded Successfully ---")
}, error = function(e) {
  stop("Error: Could not find 'header.csv' or 'box_score.csv'. 
       Make sure they are in the same directory as your R script.")
})

print("--- Step 2: Aggregating Player Stats to Team Stats... ---")
team_game_stats <- box_score %>%
  group_by(game_id, team_id) %>%
  summarise(
    pts = sum(points, na.rm = TRUE),
    fgm = sum(two_points_made, na.rm = TRUE) + sum(three_points_made, na.rm = TRUE),
    fga = sum(two_points_attempted, na.rm = TRUE) + sum(three_points_attempted, na.rm = TRUE),
    fg3m = sum(three_points_made, na.rm = TRUE),
    fg3a = sum(three_points_attempted, na.rm = TRUE),
    ftm = sum(free_throws_made, na.rm = TRUE),
    fta = sum(free_throws_attempted, na.rm = TRUE),
    reb = sum(total_rebounds, na.rm = TRUE),
    ast = sum(assists, na.rm = TRUE),
    stl = sum(steals, na.rm = TRUE),
    tov = sum(turnovers, na.rm = TRUE),
    blk = sum(blocks_favour, na.rm = TRUE),
    pir = sum(valuation, na.rm = TRUE),
    .groups = 'drop'
  )
print("--- Step 2: Aggregation Complete ---")

print("--- Step 3: Creating Rolling Average Features (Form)... ---")
header_slim <- header %>% 
  select(game_id, date, team_id_a, team_id_b, score_a, score_b) %>%
  mutate(date = as_date(date))

team_stats_with_details <- team_game_stats %>%
  left_join(header_slim, by = "game_id") %>%
  mutate(
    location = ifelse(team_id == team_id_a, "Home", "Away"),
    opponent_id = ifelse(team_id == team_id_a, team_id_b, team_id_a),
    opponent_score = ifelse(team_id == team_id_a, score_b, score_a)
  ) %>%
  select(game_id, date, team_id, location, pts, opponent_score, fgm, fga, fg3m, fg3a, ftm, fta, reb, ast, stl, tov, blk, pir) %>%
  arrange(team_id, date)

team_features <- team_stats_with_details %>%
  group_by(team_id) %>%
  mutate(
    efg_pct = (fgm + 0.5 * fg3m) / fga,
    avg_pts_last10 = lag(roll_meanr(pts, n = 10, fill = NA), 1),
    avg_pts_allowed_last10 = lag(roll_meanr(opponent_score, n = 10, fill = NA), 1),
    avg_ast_last10 = lag(roll_meanr(ast, n = 10, fill = NA), 1),
    avg_reb_last10 = lag(roll_meanr(reb, n = 10, fill = NA), 1),
    avg_tov_last10 = lag(roll_meanr(tov, n = 10, fill = NA), 1),
    avg_efg_last10 = lag(roll_meanr(efg_pct, n = 10, fill = NA), 1),
    avg_pir_last10 = lag(roll_meanr(pir, n = 10, fill = NA), 1)
  ) %>%
  ungroup() %>%
  select(game_id, team_id, date, avg_pts_last10, avg_pts_allowed_last10, 
         avg_ast_last10, avg_reb_last10, avg_tov_last10, avg_efg_last10, avg_pir_last10)
print("--- Step 3: Rolling Features Created ---")


print("--- Step 3.5: Creating H2H Features... ---")
h2h_data <- header_slim %>%
  arrange(date) %>%
  group_by(team_id_a, team_id_b) %>%
  mutate(
    game_number_h2h = row_number(),
    home_wins_h2h = lag(cumsum(score_a > score_b), 1, default = 0),
    total_games_h2h = lag(game_number_h2h, 1, default = 0),
    home_vs_away_h2h_win_pct = ifelse(total_games_h2h == 0, 0.5, home_wins_h2h / total_games_h2h)
  ) %>%
  ungroup() %>%
  select(game_id, home_vs_away_h2h_win_pct)
print("--- Step 3.5: H2H Features Created ---")


print("--- Step 4: Creating Final Modeling Dataset... ---")
model_data <- header_slim %>%
  left_join(team_features, by = c("game_id", "team_id_a" = "team_id")) %>%
  rename_with(~ paste0("home_", .), .cols = starts_with("avg_")) %>%
  left_join(team_features, by = c("game_id", "team_id_b" = "team_id")) %>%
  rename_with(~ paste0("away_", .), .cols = starts_with("avg_")) %>%
  left_join(h2h_data, by = "game_id") %>%
  mutate(
    total_points = score_a + score_b,
    home_team_won = ifelse(score_a > score_b, 1, 0),
    team_id_a = as.factor(team_id_a),
    team_id_b = as.factor(team_id_b)
  ) %>%
  na.omit() %>%
  arrange(date)

form_feature_names <- names(model_data)[grep("^home_avg_|^away_avg_", names(model_data))]
all_feature_names <- c(form_feature_names, "home_vs_away_h2h_win_pct")

split_point <- floor(nrow(model_data) * 0.8)
train_data <- model_data[1:split_point, ]
test_data <- model_data[(split_point + 1):nrow(model_data), ]
print("--- Step 4: Final Dataset Created ---")



predictor_string <- paste(all_feature_names, collapse = " + ")

win_formula_string <- paste("as.factor(home_team_won) ~", predictor_string, "+ (1 | team_id_a) + (1 | team_id_b)")
point_formula_string <- paste("total_points ~", predictor_string, "+ (1 | team_id_a) + (1 | team_id_b)")

win_formula <- as.formula(win_formula_string)
point_formula <- as.formula(point_formula_string)


print("Training Win/Lose model... (This may take a few minutes)")
win_model <- glmer(
  win_formula, # Use the explicit formula
  data = train_data, # Pass the full training data
  family = binomial # Logistic regression
)
print("Win/Lose Model Trained.")

print("Training Over/Under model... (This may also take time)")
point_model <- lmer(
  point_formula, # Use the explicit formula
  data = train_data # Pass the full training data
)
print("Over/Under Model Trained.")

print("Calculating model error (Sigma)...")
test_predictions <- predict(point_model, test_data, allow.new.levels = TRUE)
residuals <- test_data$total_points - test_predictions
model_sigma <- sd(residuals)

print(paste("Model Sigma (average point error) calculated:", round(model_sigma, 2)))
print("--- Step 5: Models are Trained and Ready ---")



get_latest_team_features <- function(team_id) {
  latest_features <- team_features %>%
    filter(team_id == !!team_id) %>%
    arrange(date) %>%
    slice_tail(n = 1) %>%
    select(starts_with("avg_"))
  
  if (nrow(latest_features) == 0) {
    stop(paste("Could not find feature data for team:", team_id))
  }
  return(latest_features)
}

calculate_current_h2h_specific <- function(home_id, away_id) {
  history <- header_slim %>% 
    filter(team_id_a == home_id, team_id_b == away_id)
  
  if (nrow(history) == 0) {
    return(0.5) # No history, return neutral 50/50
  }
  
  home_wins <- sum(history$score_a > history$score_b)
  win_pct <- home_wins / nrow(history)
  return(win_pct)
}

predict_win_lose <- function(home_id, away_id) {
  
  home_features <- get_latest_team_features(home_id) %>%
    rename_with(~ paste0("home_", .))
  
  away_features <- get_latest_team_features(away_id) %>%
    rename_with(~ paste0("away_", .))
  
  h2h_feature <- calculate_current_h2h_specific(home_id, away_id)
  
  prediction_df <- cbind(home_features, away_features)
  prediction_df$home_vs_away_h2h_win_pct <- h2h_feature
  prediction_df$team_id_a <- as.factor(home_id)
  prediction_df$team_id_b <- as.factor(away_id)
  
  win_prob_home <- predict(win_model, newdata = prediction_df, type = "response", allow.new.levels = TRUE)
  win_prob_away <- 1 - win_prob_home
  
  output <- list(
    home_team = home_id,
    away_team = away_id,
    home_win_probability = scales::percent(win_prob_home, accuracy = 0.1),
    away_win_probability = scales::percent(win_prob_away, accuracy = 0.1),
    predicted_home_decimal_odds = round(1 / win_prob_home, 2),
    predicted_away_decimal_odds = round(1 / win_prob_away, 2)
  )
  
  return(output)
}


predict_over_under <- function(home_id, away_id, line) {
  
  home_features <- get_latest_team_features(home_id) %>%
    rename_with(~ paste0("home_", .))
  
  away_features <- get_latest_team_features(away_id) %>%
    rename_with(~ paste0("away_", .))
  
  h2h_feature <- calculate_current_h2h_specific(home_id, away_id)
  
  prediction_df <- cbind(home_features, away_features)
  prediction_df$home_vs_away_h2h_win_pct <- h2h_feature
  prediction_df$team_id_a <- as.factor(home_id)
  prediction_df$team_id_b <- as.factor(away_id)
  
  predicted_mean_points <- predict(point_model, newdata = prediction_df, allow.new.levels = TRUE)
  
  prob_under <- pnorm(
    line, 
    mean = predicted_mean_points, 
    sd = model_sigma
  )
  prob_over <- 1 - prob_under
  
  output <- list(
    home_team = home_id,
    away_team = away_id,
    bookmaker_line = line,
    model_predicted_total = round(predicted_mean_points, 1),
    probability_OVER = scales::percent(prob_over, accuracy = 0.1),
    probability_UNDER = scales::percent(prob_under, accuracy = 0.1),
    predicted_OVER_odds = round(1 / prob_over, 2),
    predicted_UNDER_odds = round(1 / prob_under, 2)
  )
  
  return(output)
}

print("--- Step 6: Prediction Functions Defined ---")



if (nrow(train_data) >= 2) {
  
  example_teams <- sample(unique(train_data$team_id_a), 2)
  id_1 <- as.character(example_teams[1])
  id_2 <- as.character(example_teams[2])
  
  cat(paste("\n--- PREDICTING WIN/LOSE: ", id_1, "(Home) vs.", id_2, "(Away) ---\n"))
  
  win_odds <- predict_win_lose(home_id = id_1, away_id = id_2)
  print(win_odds)
  
  example_line <- 160.5 # Set a hypothetical line
  
  cat(paste("\n--- PREDICTING OVER/UNDER (Line:", example_line, "): ", id_1, "vs.", id_2, "---\n"))
  
  ou_odds <- predict_over_under(home_id = id_1, away_id = id_2, line = example_line)
  print(ou_odds)
  
} else {
  print("Not enough data in the training set to run an example prediction.")
}





predict_win_lose_season <- function(
    schedule_file = "Euroleague_Schedule.xlsx",
    output_xlsx_file = "Euroleague_Predictions.xlsx",
    output_csv_file = "Euroleague_Predictions.csv"
) {
  
  if(!file.exists(schedule_file)) {
    stop(paste("ERROR: Schedule Excel file not found at:", schedule_file))
  }
  
  message("Reading schedule (Excel format)...")
  
  tryCatch({
    schedule_data <- readxl::read_excel(schedule_file, sheet = 1, col_types = "guess") 
    
  }, error = function(e) {
    stop(paste("ERROR: Failed to read Excel file. Check file path and format.", e$message))
  })
  
  
  col_round <- "Round"      
  col_home  <- "HomeTeam"   
  col_away  <- "AwayTeam"   
  
  if(!(col_round %in% names(schedule_data)) | !(col_home %in% names(schedule_data)) | !(col_away %in% names(schedule_data))) {
    stop(paste("ERROR: The schedule file must contain the columns:", col_round, col_home, col_away))
  }
  
  results_list <- list()
  
  message("Predicting matches...")
  
  for(i in 1:nrow(schedule_data)) {
    
    current_round <- schedule_data[[col_round]][i]
    home_team <- as.character(schedule_data[[col_home]][i])
    away_team <- as.character(schedule_data[[col_away]][i])
    
    tryCatch({
      prediction_result <- predict_win_lose(home_team, away_team)
      
      results_list[[i]] <- data.frame(
        Round = current_round,
        HomeTeam = home_team,
        AwayTeam = away_team,
        HomeWinProb = prediction_result$home_win_probability,
        AwayWinProb = prediction_result$away_win_probability,
        HomeOdds = prediction_result$predicted_home_decimal_odds,
        AwayOdds = prediction_result$predicted_away_decimal_odds,
        stringsAsFactors = FALSE
      )
      
    }, error = function(e) {
      warning(paste("No prediction for:", home_team, "-", away_team, "Error:", e$message))
      results_list[[i]] <- data.frame(
        Round = current_round,
        HomeTeam = home_team,
        AwayTeam = away_team,
        HomeWinProb = "ERROR",
        AwayWinProb = "ERROR",
        HomeOdds = "ERROR",
        AwayOdds = "ERROR",
        stringsAsFactors = FALSE
      )
    })
  }
  
  final_results <- do.call(rbind, results_list)
  
  message("Prediction complete. Writing results to files...")
  
  tryCatch({
    writexl::write_xlsx(final_results, path = output_xlsx_file)
    message(paste("Successfully wrote results to Excel:", output_xlsx_file))
  }, error = function(e) {
    warning(paste("ERROR: Could not write to Excel file:", e$message))
  })
  
  tryCatch({
    write.csv(final_results, file = output_csv_file, row.names = FALSE)
    message(paste("Successfully wrote results to CSV:", output_csv_file))
  }, error = function(e) {
    warning(paste("ERROR: Could not write to CSV file:", e$message))
  })
  
  return(final_results)
}


INPUT_SCHEDULE_PATH <- "Euroleague_Schedule.xlsx"
OUTPUT_EXCEL_PATH <- "Euroleague_Predictions.xlsx"
OUTPUT_CSV_PATH <- "Euroleague_Predictions.csv"

season_results <- predict_win_lose_season(
  schedule_file = INPUT_SCHEDULE_PATH,
  output_xlsx_file = OUTPUT_EXCEL_PATH,
  output_csv_file = OUTPUT_CSV_PATH
)

print(head(season_results))





input_csv <- "Euroleague_Predictions.csv"

if (!file.exists(input_csv)) {
  stop("Predictions CSV not found. Please run the main prediction script first.")
}

data <- read.csv(input_csv, stringsAsFactors = FALSE)

data$HomeWinProbNum <- as.numeric(gsub("%", "", data$HomeWinProb)) / 100
data$AwayWinProbNum <- as.numeric(gsub("%", "", data$AwayWinProb)) / 100

data <- data %>% 
  filter(!is.na(HomeWinProbNum) & !is.na(AwayWinProbNum))

data$PredictedWinner <- ifelse(data$HomeWinProbNum > data$AwayWinProbNum, data$HomeTeam, data$AwayTeam)
data$WinnerProb <- pmax(data$HomeWinProbNum, data$AwayWinProbNum)
data$WinnerOdd <- ifelse(data$HomeWinProbNum > data$AwayWinProbNum, data$HomeOdds, data$AwayOdds)

data <- data %>% filter(!is.na(WinnerProb))

all_slips_list <- list()

process_and_print_slip <- function(slip_data, category_name, option_num) {
  joint_prob <- prod(slip_data$WinnerProb, na.rm = TRUE)
  total_return <- prod(slip_data$WinnerOdd, na.rm = TRUE)
  
  cat(paste0("\n--- ", category_name, " | Option ", option_num, " (Round ", slip_data$Round[1], ") ---\n"))
  cat(sprintf("%-10s %-10s %-10s %-10s %-10s %-10s %-10s\n", 
              "HomeTeam", "AwayTeam", "HomeProb", "AwayProb", "HomeOdd", "AwayOdd", "Prediction"))
  
  for(i in 1:nrow(slip_data)) {
    cat(sprintf("%-10s %-10s %-10s %-10s %-10.2f %-10.2f %-10s\n",
                slip_data$HomeTeam[i],
                slip_data$AwayTeam[i],
                slip_data$HomeWinProb[i],
                slip_data$AwayWinProb[i],
                slip_data$HomeOdds[i],
                slip_data$AwayOdds[i],
                slip_data$PredictedWinner[i]))
  }
  
  cat("\n")
  cat(sprintf("Expected return: %.3f\n", total_return))
  cat(sprintf("Expected combined probability to win: %.1f%%\n", joint_prob * 100))
  cat("------------------------------------------------------\n")
  
  slip_export <- slip_data %>%
    mutate(
      RiskCategory = category_name,
      OptionNumber = option_num,
      SlipTotalProb_Percent = round(joint_prob * 100, 2),
      SlipExpectedReturn = round(total_return, 3)
    ) %>%
    select(
      Round, RiskCategory, OptionNumber, 
      HomeTeam, AwayTeam, PredictedWinner, 
      HomeWinProb, AwayWinProb, HomeOdds, AwayOdds,
      SlipTotalProb_Percent, SlipExpectedReturn
    )
  
  return(slip_export)
}

find_top_combos_tree <- function(round_matches, target_prob, min_limit, max_limit, limit = 5) {
  
  valid_combos <- list() 
  
  decision_node <- function(idx, current_indices) {
    
    if (length(current_indices) == 3) {
      selected_probs <- round_matches$WinnerProb[current_indices]
      if (any(is.na(selected_probs))) return()
      
      joint_p <- prod(selected_probs)
      
      if (joint_p > min_limit && joint_p <= max_limit) {
        current_diff <- abs(joint_p - target_prob)
        valid_combos <<- append(valid_combos, list(list(indices=current_indices, diff=current_diff)))
      }
      return() 
    }
    
    if (idx > nrow(round_matches)) return()
    
    decision_node(idx + 1, c(current_indices, idx))
    decision_node(idx + 1, current_indices)
  }
  
  decision_node(1, c())
  
  if (length(valid_combos) == 0) return(NULL)
  
  valid_combos <- valid_combos[order(sapply(valid_combos, function(x) x$diff))]
  top_k <- head(valid_combos, limit)
  
  result_slips <- lapply(top_k, function(x) {
    round_matches[x$indices, ]
  })
  
  return(result_slips)
}

rounds <- unique(data$Round)
cat("Calculating Best Combinations...\n")

for (r in rounds) {
  round_matches <- data %>% filter(Round == r)
  if (nrow(round_matches) < 3) next
  
  low_risk_list  <- find_top_combos_tree(round_matches, 0.70, 0.60, 1.00, limit=1)
  mod_risk_list  <- find_top_combos_tree(round_matches, 0.50, 0.20, 0.60, limit=1)
  high_risk_list <- find_top_combos_tree(round_matches, 0.20, 0.00, 0.20, limit=1)
  
  if (!is.null(low_risk_list)) {
    for(k in 1:length(low_risk_list)) {
      df_chunk <- process_and_print_slip(low_risk_list[[k]], "Low Risk", k)
      all_slips_list[[length(all_slips_list) + 1]] <- df_chunk
    }
  }
  
  if (!is.null(mod_risk_list)) {
    for(k in 1:length(mod_risk_list)) {
      df_chunk <- process_and_print_slip(mod_risk_list[[k]], "Moderate Risk", k)
      all_slips_list[[length(all_slips_list) + 1]] <- df_chunk
    }
  }
  
  if (!is.null(high_risk_list)) {
    for(k in 1:length(high_risk_list)) {
      df_chunk <- process_and_print_slip(high_risk_list[[k]], "High Risk", k)
      all_slips_list[[length(all_slips_list) + 1]] <- df_chunk
    }
  }
}

if (length(all_slips_list) > 0) {
  cat("\nWriting results to 'betslips.xlsx'...\n")
  
  final_betslip_df <- do.call(rbind, all_slips_list)
  
  output_file <- "betslips.xlsx"
  
  tryCatch({
    write_xlsx(final_betslip_df, path = output_file)
    message(paste("SUCCESS: Betting slips saved to", output_file))
  }, error = function(e) {
    message(paste("ERROR writing Excel file:", e$message))
  })
  
} else {
  message("No valid betting slips found to write.")
}





print("\n--- Evaluating Win/Lose Model Performance (Threshold = 0.7) ---\n")

y_true <- test_data$home_team_won

y_prob <- predict(win_model, newdata = test_data, type = "response", allow.new.levels = TRUE)

y_pred <- ifelse(y_prob >= 0.7, 1, 0)

conf_matrix <- table(
  Actual = y_true,
  Predicted = y_pred
)

print("Confusion Matrix:")
print(conf_matrix)

accuracy <- mean(y_pred == y_true)

precision <- sum(y_pred == 1 & y_true == 1) / sum(y_pred == 1)

recall <- sum(y_pred == 1 & y_true == 1) / sum(y_true == 1)

if (!require(pROC)) install.packages("pROC")
library(pROC)
auc_value <- auc(y_true, y_prob)

cat("\n--- MODEL PERFORMANCE with Threshold 0.7 ---\n")
cat("Accuracy :", round(accuracy, 4), "\n")
cat("Precision:", round(precision, 4), "\n")
cat("Recall   :", round(recall, 4), "\n")
cat("AUC      :", round(auc_value, 4), "\n")
cat("--------------------------------------\n")


install.packages("Metrics")



library(Metrics)   # RMSE, MAE
library(forecast)  # MASE

y_true <- test_data$home_team_won

y_prob <- predict(win_model, newdata = test_data, type = "response", allow.new.levels = TRUE)

rmse_val <- rmse(y_true, y_prob)

mae_val <- mae(y_true, y_prob)

mpe_val <- mean((y_true - y_prob) / ifelse(y_true==0, 1, y_true)) * 100

mape_val <- mean(abs(y_true - y_prob) / ifelse(y_true==0, 1, y_true)) * 100

y_naive <- lag(y_true, 1)
y_naive[is.na(y_naive)] <- mean(y_true) # ilk değer için ortalama kullan
mase_val <- mean(abs(y_true - y_prob)) / mean(abs(y_true - y_naive))

residuals_prob <- y_true - y_prob
acf1_val <- acf(residuals_prob, lag.max = 1, plot = FALSE)$acf[2]

cat("\n--- WIN/LOSE MODEL PERFORMANCE METRICS ---\n")
cat(sprintf("RMSE  : %.4f\n", rmse_val))
cat(sprintf("MAE   : %.4f\n", mae_val))
cat(sprintf("MPE   : %.2f%%\n", mpe_val))
cat(sprintf("MAPE  : %.2f%%\n", mape_val))
cat(sprintf("MASE  : %.4f\n", mase_val))
cat(sprintf("ACF1  : %.4f\n", acf1_val))
cat("-----------------------------------------\n")
