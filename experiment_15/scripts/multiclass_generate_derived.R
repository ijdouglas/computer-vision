# The results contain columns: "img", "notJA_prob", "JA_prob"
# Read in the test set for each subject and merge with results
# First cmd line arg must be directory to the results
# Second argument is directory with the corresponding test sets
# The script overwrites the results after merging in ground truth
# The third argument must be the directory in which to save the derived results
args = commandArgs(trailingOnly=TRUE)
if (length(args) == 0) stop("path to the results dir must be provided as first argument")
if (length(args) == 1) stop("path to the results dir must be the first argument, path to the test sets dir must be second argument")
if (length(args) == 2) stop("path to the results dir must be the first argument, path to the test sets dir must be second argument, the third argument must be the directory in which to save the derived results")
print(paste0("\n Starting script at: ", Sys.time()))
library(stringr)
library(dplyr)
library(tidyr)
library(parallel)
library(purrr)
library(MLmetrics)
library(readr)
if (length(args) == 3) {	
  results = list.files(args[1]) # just the file names
  names(results) <- results
  final_results = mclapply(results, mc.cores = 10, FUN= function(r) {
      file = file.path(args[1], r)
      res = read.csv(file, stringsAsFactors = F)
      test_file = str_replace(r, 'results', 'test')
      test = read.csv(file.path(args[2], test_file), stringsAsFactors = F)
      # The following will be returned and written out:
      res %>% rowwise() %>%
	mutate(img = as.numeric(tail(str_extract_all(img, '[[:digit:]]+')[[1]], 1))) %>%
        ungroup %>%
	rename(frame = img, notJA_prob = class_0, child_led_prob = class_1, parent_led_prob = class_2) %>%
        left_join(test, by = 'frame') %>%
	# code derived variables based on results and original data
	rowwise() %>%
	mutate(prediction = which.max(c(notJA_prob, child_led_prob, parent_led_prob)) - 1) %>%
	ungroup %>%
	mutate(correct = leader == prediction,
	       parent_eye_child_face = cstream_eye_roi_fixation_parent==11)
  })
  # Write out derived results
  iwalk(final_results, ~{
    derived_filename = str_replace(.y, 'results', 'derived')
    write.csv(.x, file.path(args[3], derived_filename), row.names=F)
  })
}
print(paste0("\nScript finished at ", Sys.time())) 
