# The results contain columns: "img", "notJA_prob", "JA_prob"
# Read in the test set for each subject and merge with results
# First cmd line arg must be directory to the results
# Second argument is directory with the corresponding test sets
# The script overwrites the results after merging in ground truth
args = commandArgs(trailingOnly=TRUE)
if (length(args) == 0) stop("path to the results dir must be provided as first argument")
if (length(args) == 1) stop("path to the results dir must be the first argument, path to the test sets dir must be second argument")
library(stringr)
library(dplyr)
library(tidyr)
library(parallel)
library(purrr)
if (length(args) == 2) {	
  results = list.files(args[1])
  names(results) <- results
  final_results = mclapply(results, mc.cores = 10, FUN= function(r) {
      file = file.path(args[1], r)
      res = read.csv(file, stringsAsFactors = F)
      test_file = str_replace(r, 'results', 'test')
      test = read.csv(file.path(args[2], test_file), stringsAsFactors = F)
      out = res %>% rowwise() %>%
	mutate(frame = as.numeric(tail(str_extract_all(img, '[[:digit:]]+')[[1]], 1))) %>%
        ungroup %>%
        left_join(test, by = 'frame')	
  })
  iwalk(final_results, ~{
    write.csv(.x, file.path(args[1], .y), row.names=F)
  })
}
