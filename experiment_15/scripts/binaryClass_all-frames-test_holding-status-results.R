# Read in the data frame of cont variables
# Read in the derived results
# Merge them
# Partition results and recompute accuracy/confusion matrices
# for different holding scenarios: child only holds visible object, parent only, both, neither.
# These are calculated on ground truth JA only
args = commandArgs(trailingOnly = T)
who = args[1]
cfile = args[2]
if (length(args) > 2) pfile = args[3]
#if (who == 'both') {
#	print(paste0('Saving child results to: ', cfile))
#	print(paste0('Saving parent results to: ', pfile))
#	stopifnot(file.exists(cfile))
#	stopifnot(file.exists(pfile))
#} else if (who == 'child') {
#	print(paste0('Saving child results to: ', cfile))
#	stopifnot(file.exists(cfile))
#} else if (who == 'parent') {
#	print(paste0('Saving parent results to: ', pfile))
#	stopifnot(file.exists(pfile))
#}
source('functions/process_derived.R')
cder = merge_results("../derived/derived_results/all_frames_test_pct20_train/child")
pder = merge_results("../derived/derived_results/all_frames_test_pct20_train/parent")
ccont = read.csv("../data/master/all-child-all-subjects-all-obj_cont-obj-size.csv", stringsAsFactors=F)
pcont = read.csv("../data/master/all-parent-all-subjects-all-obj_cont-obj-size.csv", stringsAsFactors=F)
cdata = left_join(cder %>% 
		    arrange(subject, frame) %>%
		    select(subject, kidID, frame, joint_attention, leader, target, 
		           JA_prob, notJA_prob, prediction, 
			   ends_with('face'), starts_with('cstream_inhand')), 
		  ccont, by = c('subject', 'frame'))
pdata = left_join(pder %>% 
		    arrange(subject, frame) %>%
		    select(subject, kidID, frame, joint_attention, leader, target,
			   JA_prob, notJA_prob, prediction, 
			   ends_with('face'), starts_with('cstream_inhand')),
                   pcont, by = c('subject', 'frame'))
ccont_matrix = as.matrix(cdata %>% select(starts_with('cont_vision')))
pcont_matrix = as.matrix(pdata %>% select(starts_with('cont_vision')))
get_value = function(mat, idx, cstream) {
  if (is.na(cstream[idx])) {
    out = NA
  } else if (cstream[idx] == 0) {
    out = 0
  } else {
    out = mat[idx, cstream[idx]]
  }
  return(out)
}
cdata$visible_inhand_left_child = unlist(sapply(1:nrow(cdata), function(i) {
  get_value(ccont_matrix, i, cdata$cstream_inhand_left_hand_obj_all_child)
}))
cdata$visible_inhand_right_child = unlist(sapply(1:nrow(cdata), function(i) {
  get_value(ccont_matrix, i, cdata$cstream_inhand_right_hand_obj_all_child)
}))
cdata$visible_inhand_left_parent = unlist(sapply(1:nrow(cdata), function(i) {
  get_value(ccont_matrix, i, cdata$cstream_inhand_left_hand_obj_all_parent)
}))
cdata$visible_inhand_right_parent = unlist(sapply(1:nrow(cdata), function(i) {
  get_value(ccont_matrix, i, cdata$cstream_inhand_right_hand_obj_all_parent)
}))
#cdata = cdata %>% mutate(across(starts_with('visible_inhand'), ~ifelse(is.na(.), 0, .)))
pdata$visible_inhand_left_child = unlist(sapply(1:nrow(pdata), function(i) {
  get_value(pcont_matrix, i, pdata$cstream_inhand_left_hand_obj_all_child)
}))
pdata$visible_inhand_right_child = unlist(sapply(1:nrow(pdata), function(i) {
  get_value(pcont_matrix, i, pdata$cstream_inhand_right_hand_obj_all_child)
}))
pdata$visible_inhand_left_parent = unlist(sapply(1:nrow(pdata), function(i) {
  get_value(pcont_matrix, i, pdata$cstream_inhand_left_hand_obj_all_parent)
}))
pdata$visible_inhand_right_parent = unlist(sapply(1:nrow(pdata), function(i) {
  get_value(pcont_matrix, i, pdata$cstream_inhand_right_hand_obj_all_parent)
}))
#pdata = pdata %>% mutate(across(starts_with('visible_inhand'), ~ifelse(is.na(.), 0, .)))
# Code the 4 holding statuses
cdata = cdata %>% mutate(visible_inhand_viewer = as.numeric((visible_inhand_right_child + visible_inhand_left_child) > 0),
			 visible_inhand_partner = as.numeric((visible_inhand_right_parent + visible_inhand_left_parent) > 0),
			 visible_inhand_both = as.numeric((visible_inhand_viewer + visible_inhand_partner) == 2),
			 visible_inhand_neither = as.numeric((visible_inhand_viewer + visible_inhand_partner) == 0))
pdata = pdata %>% mutate(visible_inhand_partner = as.numeric((visible_inhand_right_child + visible_inhand_left_child) > 0),
                         visible_inhand_viewer = as.numeric((visible_inhand_right_parent + visible_inhand_left_parent) > 0),
                         visible_inhand_both = as.numeric((visible_inhand_partner + visible_inhand_viewer) == 2),
                         visible_inhand_neither = as.numeric((visible_inhand_partner + visible_inhand_viewer) == 0))
write.csv(cdata, file.path(cfile, 'holding_status.csv'), row.names=F)
write.csv(pdata, file.path(pfile, 'holding_status.csv'), row.names = F)

