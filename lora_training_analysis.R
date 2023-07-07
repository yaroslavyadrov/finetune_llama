# Load libraries
library(jsonlite)
library(stringr)
library(tidyverse)

mid_y <-  0.80
width_y <- 0.20
label_epochs <- 1

model_names <-
  c(
    "open_llama_13b-wizardlm_r4_100000000",
    "llama_13b-wizardlm-100000000",
    "open_llama_13b-wizardlm-100000000"
    # "open_llama_7b-oasst1-10713050"
    # "open_llama_7b-baize-alp_so_qra-10161196",
    # "open_llama_7b-baize-alp_so_qra-10036835",
    # "llama-7b-baize-alp_so_qra-7016907"
  )

data_dir <- "/data/lora/finetuned_models/"

###########################################################

copy_latest_checkpoints <-
  function(model_names) {
    for (model_name in model_names) {
      # Copy from the model directory
      model_dir <- paste0(data_dir, model_name)
      destination_file <- paste0(model_name, "_params.txt")

      if (!file.exists(destination_file)) {
        system(
          paste0(
            "scp asushimu:/home/mulderg/Work/baize-chatbot/train_lora.sh ",
            destination_file
          )
        )
      } else {
        print(paste0(destination_file, " already exists, not overwriting"))
      }

      # Find the most recent checkpoint directory for this model
      most_recent_checkpoint <-
        system(paste0("ssh asushimu 'ls -dtr ", model_dir, "/checkpoint*' | tail -n 1"),
               intern = TRUE)

      # Copy the trainer_state.json file from the most recent checkpoint directory
      # Use the model name to differentiate between the files
      destination_file <- paste0(model_name, "_trainer_state.json")
      # unlink(destination_file)
      system(
        paste0(
          "rsync -avz asushimu:",
          most_recent_checkpoint,
          "/trainer_state.json ",
          destination_file
        )
      )
    }
  }

get_json_data <-
  function(model_names) {
    # Initialize data frames
    loss_df <- tibble()
    eval_loss_df <- tibble()

    for (model_name in model_names) {
      # File name to be read
      trainer_state_file <-
        paste0(model_name, "_", "trainer_state.json")

      # Parse JSON into a list
      logs <-
        jsonlite::fromJSON(trainer_state_file, simplifyVector = FALSE)

      # Convert log history to a data frame
      for (log in logs$log_history) {
        if ("loss" %in% names(log) && is.numeric(log$loss)) {
          row <- tibble(
            model = model_name,
            loss = log$loss,
            epoch = log$epoch,
            step = log$step
          )
          loss_df <- bind_rows(loss_df, row)
        }

        if ("eval_loss" %in% names(log) &&
            is.numeric(log$eval_loss)) {
          row <- tibble(
            model = model_name,
            loss = log$eval_loss,
            epoch = log$epoch,
            step = log$step
          )
          eval_loss_df <- bind_rows(eval_loss_df, row)
        }
      }
    }
    return(list(loss_df = loss_df, eval_loss_df = eval_loss_df))
  }

extract_bash_vars <- function(file_path) {
  # Read the file
  lines <- readLines(file_path)

  # Concatenate all lines into a single string
  full_text <- paste(lines, collapse = " ")

  # Match Bash variable declarations using regular expressions
  # This matches variables like VAR=value or VAR="value"
  matches <-
    gregexpr("\\b[A-Z_0-9]+=([\"']?)[^ \"]*\\1\\b", full_text, perl = TRUE)

  # Extract the matches
  vars <- regmatches(full_text, matches)

  # Remove any empty matches
  vars <- Filter(nchar, unlist(vars))

  # Split the variable names and values
  var_split <- strsplit(vars, "=", fixed = TRUE)

  # Create a named list
  var_list <-
    setNames(lapply(var_split, `[`, 2), sapply(var_split, `[`, 1))

  return(var_list)
}

parse_params <- function(file_path) {
  # Read the file
  lines <- readLines(file_path)

  # Initialize an empty list to store the parameters
  params <- list()

  # Iterate over each line
  default_warn <- getOption("warn")
  options(warn = -1)
  for (line in lines) {
    # Use a regular expression to match '--parameter value' format
    matches <- gregexpr("--([a-z_]+) ([^ ]+)", line, perl = TRUE)
    matches_data <- regmatches(line, matches)[[1]]

    # Iterate over the matched pairs
    for (pair in matches_data) {
      # Split the pair into parameter and value
      pair_split <- strsplit(pair, " ")[[1]]
      param <- pair_split[1]
      value <- pair_split[2]

      # Remove the '--' from the parameter name
      param <- substr(param, 3, nchar(param))

      # Add the parameter and value to the list
      # Convert numeric values to numeric type
      if (is.na(as.numeric(value))) {
        params[[param]] <- value
      } else {
        params[[param]] <- as.numeric(value)
      }
    }
  }
  options(warn = default_warn)

  # Return the list of parameters
  return(params)
}

get_model_params <- function(model_names) {
  hyper_params <- tibble()

  for (model_name in model_names) {
    # File name to be read
    file_name <- paste0(model_name, "_params.txt")

    bash_vars <- extract_bash_vars(file_name)
    python_params <- parse_params(file_name)

    tibble_row <-
      tibble(!!!c(bash_vars, python_params)) %>%
      mutate(model = model_name)

    # Append tibble to the main hyper_params tibble
    hyper_params <- bind_rows(hyper_params, tibble_row)
  }

  return(hyper_params)
}


copy_latest_checkpoints(model_names)
losses <- get_json_data(model_names)

# Create a column for the loss type
losses[["loss_df"]]$type <- "train"
losses[["eval_loss_df"]]$type <- "eval"

max_epoch <-
  losses[["eval_loss_df"]] %>%
  filter(model == model_names[1]) %>%
  pull(epoch) %>%
  max

max_steps <-
  losses[["eval_loss_df"]] %>%
  filter(model == model_names[1]) %>%
  pull(step) %>%
  max

loss_data <-
  bind_rows(losses[["eval_loss_df"]], losses[["loss_df"]]) %>%
  # filter(epoch >= 0.20 & epoch <= max_epoch) %>%
  filter(epoch <= 1.0) %>%
  # filter(step >= 100 & step <= max_steps) %>%
  group_by(substr(model, 1, 18)) %>%
  # filter(loss < quantile(loss, 0.75)) %>%
  mutate(scaled.loss = scale(loss)) %>%
  ungroup %>%
  mutate(label = ifelse((type == "eval") &
                          (step %% 200 == 0), as.character(round(epoch, 1)), ""))

ggplot(loss_data, aes(x = epoch, y = loss, color = type)) +
  facet_wrap(~ model, ncol = 3) +
  geom_smooth(linewidth = 0.5) +
  geom_point(size = 0.5) +
  # geom_text(
  #   aes(label = label),
  #   vjust = +3,
  #   size = 3,
  #   colour = "black"
  # ) +
  labs(
    title = "Loss and Eval_Loss",
    x = "Epoch",
    y = "Loss",
    color = "Type"
  )

# hyper_params <-
#   get_model_params(model_names) %>%
#   # select_if( ~ length(unique(.)) != 1) %>%
#   select(
#     model,
#     learning_rate,
#     lora_alpha,
#     lora_dropout,
#     lora_r,
#     micro_batch_size,
#     target_modules
#   )

results <-
  loss_data %>%
  mutate(run = as.numeric(str_extract(model, "\\d+$"))) %>%
  mutate(model.data = str_remove(model, "-\\d+$")) %>%
  mutate(loss = round(loss, 3)) %>%
  select(model, model.data, type, run, step, epoch, loss) %>%
  # inner_join(hyper_params) %>%
  select(-model)

results %>% filter(type == "train") %>% arrange(step, run) %>% tail(length(model_names) * 4)

results %>% filter(type == "eval") %>% arrange(step, run) %>% tail(length(model_names) * 4) # %>% select(model.data, step, epoch, loss)
