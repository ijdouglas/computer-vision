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
who = args[3]
if (grepl('child', who)) {frame_dir = '/cam07_frames_p/'} else {frame_dir = '/cam08_frames_p/'}
train_dir = args[4]
test_dir = args[5]
master_dir = args[6]
if (!'run' %in% args) {
	stop("This is a dry run. The first argument must be the fraction by which to downsample the train data. The second argument must be either TRUE or FALSE indicating if to save out test data. The third argument must be whether to process the child or parent data (must be 'child' or 'parent'. The fourth argument must be the directory in which to save the generated train data. The fifth argument must be the directory in which to save the generated test data (simply ignored if arg 2 is FALSE but needs a dummy value nonetheless). The sixth argument must be the path to the frames_master.csv file (absolute or relative), generated from the matlab script that extracted multiwork variables. The seventh argument must be 'run' in order to execute the script")
}
print(paste0('Running script at: ', Sys.time()))
print(paste0('Running script from: ', here()))
print(paste0('Looking for master frames file in: ', master_dir))
print(paste0('Write out training data to: ', train_dir))
print(paste0('Write out test data to (optional): ', test_dir))
print(paste0('Process data for: ', who))
print(paste0('Sampling ', frac*100, ' percent of the data'))
print(paste0('Generating test data: ', make_test))
set.seed(79847)
fm = read_csv(master_dir)
# Convert to data tables rather than tibbles
fm = lazy_dt(fm)
# For each subject, extract frames associated with trials
tic()
fm = fm %>%
	filter(cstream_trials > 0) %>%
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
saveRDS(fm, "../data/paths_and_frames_master.rds")
