# First argument must be the percent of frames by which to downsample training data
# Second argument is whether or not to write out the test data
# Usage:
# `Rscript --vanilla THIS_SCRIPT.R .2 FALSE &`
# This version extracts balanced train and test after filtering the data as follows:
# 1. Extract frames appearing during experiment trials
# 2. joint attention must be defined
# 3. image must exist on multiwork
# Filter not implemented:
# a. Extracting frames ocurring during 'looks' to objects
# b. Deleting frames during looks off screen
# c. Deleting frames over which merging ocurred to produce contiguous object looks where subject looked away briefly
suppressPackageStartupMessages(library(dtplyr))
suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(tidyr))
suppressPackageStartupMessages(library(purrr))
suppressPackageStartupMessages(library(parallel))
suppressPackageStartupMessages(library(tictoc))
suppressPackageStartupMessages(library(here))
suppressPackageStartupMessages(library(data.table))
args = commandArgs(trailingOnly=TRUE)
frac = as.numeric(args[1])
make_test = as.logical(args[2])
who = args[3]
if (grepl('child', who)) {frame_dir = '/cam07_frames_p/'} else {frame_dir = '/cam08_frames_p/'}
train_dir = args[4]
test_dir = args[5]
master_file = args[6]
if (!'run' %in% args) {
	stop("This is a dry run. The first argument must be the fraction by which to downsample the train data. The second argument must be either TRUE or FALSE indicating if to save out test data. The third argument must be whether to process the child or parent data (must be 'child' or 'parent'. The fourth argument must be the directory in which to save the generated train data. The fifth argument must be the directory in which to save the generated test data (simply ignored if arg 2 is FALSE but needs a dummy value nonetheless). The sixth argument must be the path to the frames_master.csv file (absolute or relative), generated from the matlab script that extracted multiwork variables. The seventh argument must be 'run' in order to execute the script")
}
print(paste0('Running script at: ', Sys.time()))
print(paste0('Running script from: ', here()))
print(paste0('Looking for master frames file at: ', master_file))
print(paste0('Write out training data to: ', train_dir))
print(paste0('Write out test data to (optional): ', test_dir))
print(paste0('Process data for: ', who))
print(paste0('Sampling ', frac*100, ' percent of the data'))
print(paste0('Generating test data: ', make_test))
set.seed(79847)
#fm = readRDS("../data/paths_and_frames_master.rds")
fm = read_csv(master_file)
# Convert to data tables rather than tibbles
fm = lazy_dt(fm)
# For each subject, extract frames associated with trials
tic()
#fm = fm %>%
#	filter(cstream_trials > 0) %>%
#	# Just in case, make sure images that exist remain
#	filter(!is.na(image)) %>%
#	# Now drop subjects for whom no JA frames remain
#	group_by(subject) %>%
#	filter(sum(JA, na.rm=T) > 0) %>%
#	ungroup %>%
#	# Now construct the path to the images, call it path
#	# Note this is hard coded for child/parent consistent with filename of this script
#	mutate(path = paste0('experiment_', experiment, '/included/', kidID, frame_dir, image)) %>%
#	# Now also rename JA to 'joint_attention' so it is compatible with dataloader.py
#	rename(joint_attention = JA) %>%
#	# FOR MULTICLASS:
#	# Create the labels for: not joint attention, child led joint attention, parent led joint attention
#	# ALSO: drop frames where leader is 'none' (simultaneous attention switches into JA with no leader)
#	filter(leader != 'none') %>%
#	mutate(across(leader, ~case_when(. == 'not_JA' ~ 0, . == 'child' ~ 1, . == 'parent' ~ 2))) %>%
#	# As a final check (though this should be unnecessary) make sure joint_attention and leader are not NA
#	filter(!is.na(joint_attention), !is.na(leader)) %>%
#	# LASTLY: make sure that the frame jpg exists at /data/drives/multiwork
#	filter(file.exists(paste0('/data/drives/multiwork/', path)))
#fm = as_tibble(fm)
fm = readRDS("../multiclass_fm.rds")
#saveRDS(fm, "../multiclass_fm.rds")
#print(dim(fm))
toc()
# Now loop through (remaining) subjects and create/save balanced train/test
mclapply(X=unique(fm$subject), mc.cores = 10, FUN=function(id) {
    test = fm %>% filter(subject == id)
    train = fm %>% filter(subject != id)
    # Balance each within subjects
    # this requires deprecated function `dplyr::sample_n`
    # Found a solution using pluck and slice_sample
    if (make_test) {
      test = test %>% group_by(leader) %>%
	      slice_sample(n = min(table(pluck(., 'leader')))) %>%
	      ungroup()
      # write out test
      write.csv(test, paste0(file.path(test_dir, id), '_test.csv'), row.names=F)
    }
    train = map(unique(train$subject), ~{
      train %>%
	      filter(subject == .x) %>%
	      group_by(leader) %>%
	      slice_sample(n = min(table(pluck(., 'leader')))) %>%
	      ungroup
    }) %>%
    reduce(rbind) %>%
    arrange(subject,frame)
    # write out train
    write.csv(train, paste0(file.path(train_dir, id), '_train.csv'), row.names=F)
})
# Also save a list of remaining subjects
write.csv(tibble('subject' = unique(fm$subject)), '../data/subjects_to_model.csv', row.names=F)
