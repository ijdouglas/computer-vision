# This version extracts balanced train and test after filtering the data as follows:
# 1. Extract frames appearing during experiment trials
# 2. joint attention must be defined
# 3. image must exist on multiwork
# Filter not implemented:
# a. Extracting frames ocurring during 'looks' to objects
# b. Deleting frames during looks off screen
# c. Deleting frames over which merging ocurred to produce contiguous object looks where subject looked away briefly
library(dtplyr)
library(readr)
library(dplyr)
library(tidyr)
library(parallel)
library(tictoc)
set.seed(79847)
fm = read_csv("../data/frames_master.csv")
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
	mutate(path = paste0('experiment_', experiment, '/included/', kidID, '/cam08_frames_p/', image)) %>%
	# Now also rename JA to 'joint_attention' so it is compatible with dataloader.py
	rename(joint_attention = JA) %>%
	# As a final check (though this should be unnecessary) make sure joint_attention is not NA
	filter(!is.na(joint_attention)) %>%
	# LASTLY: make sure that the frame jpg exists at /data/drives/multiwork
	filter(file.exists(paste0('/data/drives/multiwork/', path)))
fm = as_tibble(fm)
toc()
# Now loop through (remaining) subjects and create/save balanced train/test
mclapply(X=unique(fm$subject), mc.cores = 10, FUN=function(id) {
    test = fm %>% filter(subject == id)
    train = fm %>% filter(subject != id)
    # Balance each within subjects
    # this requires deprecated function `dplyr::sample_n`
    test = test %>% group_by(subject) %>%
	    mutate(n_JA = sum(joint_attention, na.rm=T), n_not_JA = sum(not_JA, na.rm=T)) %>%
	    group_by(subject, joint_attention) %>%
	    sample_n(min(c(unique(n_JA), unique(n_not_JA)))) %>%
	    ungroup()
    train = train %>% group_by(subject) %>%
	    mutate(n_JA = sum(joint_attention, na.rm=T), n_not_JA = sum(not_JA, na.rm=T)) %>%
	    group_by(subject, joint_attention) %>%
	    sample_n(min(c(unique(n_JA), unique(n_not_JA)))) %>%
	    ungroup()
    # write out each data frame
    write.csv(train, paste0('../data/parent/train/', id, '_train.csv'), row.names=F)
    write.csv(test, paste0('../data/parent/test/', id, '_test.csv'), row.names=F)
})
