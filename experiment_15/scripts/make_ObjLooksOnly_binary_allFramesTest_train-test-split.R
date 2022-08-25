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
suppressPackageStartupMessages(library(parallel))
suppressPackageStartupMessages(library(tictoc))
suppressPackageStartupMessages(library(here))
suppressPackageStartupMessages(library(data.table))
args = commandArgs(trailingOnly=TRUE)
frac = as.numeric(args[1])
make_test = as.logical(args[2])
make_train = as.logical(args[3])
who = args[4]
if (grepl('child', who)) {frame_dir = '/cam07_frames_p/'} else {frame_dir = '/cam08_frames_p/'}
train_dir = args[5]
test_dir = args[6]
master_file = args[7]
num_obj = args[10]
if (!'run' %in% args) {
	stop("This is a dry run. The first argument must be the fraction by which to downsample the train data. The second argument must be either TRUE or FALSE indicating if to save out test data. The third arg must be TRUE or FALSE indicating if to save out the train data. The fourth argument must be whether to process the child or parent data (must be 'child' or 'parent'. The fifth argument must be the directory in which to save the generated train data (ignored if arg 3 is FALSE). The sixth argument must be the directory in which to save the generated test data (simply ignored if arg 2 is FALSE but needs a dummy value nonetheless). The seventh argument must be the path to the frames_master.csv file (absolute or relative), generated from the matlab script that extracted multiwork variables. The either argument is the absolute path (path and filename) for the subjects_to_model file. The ninth argument is T or F, whether to save the subjects_to_model file. The tenth argument is the number of ROI objects for this study. The eleventh argument must be 'run' in order to execute the script")
}
print(paste0('Running script at: ', Sys.time()))
print(paste0('Running script from: ', here()))
print(paste0('Looking for master frames file at: ', master_file))
print(paste0('Write out training data to: ', train_dir))
print(paste0('Write out test data to (optional): ', test_dir))
print(paste0('Process data for: ', who))
print(paste0('Sampling ', frac*100, ' percent of the data'))
print(paste0('the number of objects in this study: ', num_obj))
print(paste0('Generating test data: ', make_test))
print(paste0('Subjects-to-model file would be: ', args[8]))
print(paste0('Save subjects-to-model files?  ', args[9]))
set.seed(79847)
#fm = readRDS("../data/paths_and_frames_master.rds")
fm = read_csv(master_file)
# Convert to data tables rather than tibbles
fm = lazy_dt(fm)
# For each subject, extract frames associated with trials
tic()
fm = fm %>%
	filter(cstream_trials > 0) %>% # just use data in trials
	# Just in case, make sure images that exist remain
	filter(!is.na(image)) %>%
	# No drop subjects for whom no JA frames remain
	group_by(subject) %>%
	filter(sum(JA, na.rm=T) > 0) %>%
	ungroup %>%
	# Now construct the path to the images, call it path
	# Note this is hard coded for child/parent consistent with filename of this script
	mutate(path = paste0('experiment_', experiment, '/included/', kidID, frame_dir, image)) %>%
	# Now also rename JA to 'joint_attention' so it is compatible with dataloader.py
	rename(joint_attention = JA) %>%
	# As a final check (though this should be unnecessary) make sure joint_attention is not NA
	filter(!is.na(joint_attention)) %>%
	# LASTLY: make sure that the frame jpg exists at /data/drives/multiwork
	filter(file.exists(paste0('/data/drives/multiwork/', path)))
fm = as_tibble(fm)
#print(dim(fm))
toc()
# Now loop through (remaining) subjects and create/save balanced train
# Test is all frames for the subject since this scripts is for all-frames testing
mclapply(X=unique(fm$subject), mc.cores = 10, FUN=function(id) {
    test = fm %>% filter(subject == id)
    train = fm %>% filter(subject != id)
    # Crucialy, for this pipeline extract looks to eye ROIs only
    if (grepl('child', who)) {
      test = test %>% filter(cstream_eye_roi_fixation_child %in% 1:num_obj)
      train = train %>% filter(cstream_eye_roi_fixation_child %in% 1:num_obj)
    } else {
      test = test %>% filter(cstream_eye_roi_fixation_parent %in% 1:num_obj)
      train = train %>% filter(cstream_eye_roi_fixation_parent %in% 1:num_obj)
    }
    # Balance train within subjects
    # this requires deprecated function `dplyr::sample_n`
    if (make_test) {
      #test = test %>% group_by(subject) %>%
      #      mutate(n_JA = sum(joint_attention, na.rm=T), n_not_JA = sum(not_JA, na.rm=T)) %>%
      #	      group_by(subject, joint_attention) %>%
      #	      sample_n(min(c(unique(n_JA), unique(n_not_JA)))) %>%
      #	      ungroup()
      # write out test
      write.csv(test, paste0(file.path(test_dir, id), '_test.csv'), row.names=F)
    }
    if (make_train) {

    train = train %>% group_by(subject) %>%
	    mutate(n_JA = sum(joint_attention, na.rm=T), n_not_JA = sum(not_JA, na.rm=T)) %>%
	    group_by(subject, joint_attention) %>%
	    sample_n(min(c(unique(n_JA), unique(n_not_JA)))) %>%
	    ungroup() %>%
	    # Now downsample according to the command line option
	    as.data.table %>% 
	    data.table:::split.data.table(by='joint_attention') %>% 
	    purrr::map(slice_sample, prop = frac) %>%
	    purrr::map(as.data.frame) %>%
	    purrr::reduce(rbind) %>%
	    arrange(subject, frame)
    # write out train
    write.csv(train, paste0(file.path(train_dir, id), '_train.csv'), row.names=F)
    }
})
# Also save a list of remaining subjects
subjects_file = args[8]
if (args[9]) write.csv(tibble('subject' = unique(fm$subject)), subjects_file, row.names=F)
