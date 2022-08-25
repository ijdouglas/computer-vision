# Read in the data frame of cont variables
# Read in the derived results
# Merge them
# Partition results and recompute accuracy/confusion matrices
# for different holding scenarios: child only holds visible object, parent only, both, neither.
# These are calculated on ground truth JA only
args = commandArgs(trailingOnly = T)
cfile = args[1]
pfile = args[2]
print(paste0('Saving child results to: ', cfile))
print(paste0('Saving parent results to: ', pfile))
stopifnot(file.exists(cfile))
stopifnot(file.exists(pfile))
source('functions/process_derived.R')
cder = merge_results("../derived/child/pct20")
pder = merge_results("../derived/parent/pct20")
ccont = read.csv("../data/all-child-all-subjects-all-obj_cont-obj-size.csv", stringsAsFactors=F)
pcont = read.csv("../data/all-parent-all-subjects-all-obj_cont-obj-size.csv", stringsAsFactors=F)
cdata = left_join(cder %>% select(subject, frame, joint_attention, leader, target, 
				   JA_prob, notJA_prob, prediction, starts_with('cstream_inhand')), 
		   ccont, by = c('subject', 'frame'))
pdata = left_join(pder %>% select(subject, frame, joint_attention, leader, target,
                                   JA_prob, notJA_prob, prediction, starts_with('cstream_inhand')),
                   pcont, by = c('subject', 'frame'))
ccont_matrix = as.matrix(cdata %>% select(starts_with('cont_vision')))
pcont_matrix = as.matrix(pdata %>% select(starts_with('cont_vision')))
cdata$visible_inhand_left_child = as.numeric(sapply(1:nrow(cdata), function(i) {
  ccont_matrix[i, cdata$cstream_inhand_left_hand_obj_all_child[i]] > 0
}))
cdata$visible_inhand_right_child = as.numeric(sapply(1:nrow(cdata), function(i) {
  ccont_matrix[i, cdata$cstream_inhand_right_hand_obj_all_child[i]] > 0
}))
cdata$visible_inhand_left_parent = as.numeric(sapply(1:nrow(cdata), function(i) {
  ccont_matrix[i, cdata$cstream_inhand_left_hand_obj_all_parent[i]] > 0
}))
cdata$visible_inhand_right_parent = as.numeric(sapply(1:nrow(cdata), function(i) {
  ccont_matrix[i,cdata$stream_inhand_right_hand_obj_all_parent[i]] > 0
}))
cdata = cdata %>% mutate(across(starts_with('visible_inhand'), ~ifelse(is.na(.), 0, .)))
pdata$visible_inhand_left_child = as.numeric(sapply(1:nrow(pdata), function(i) {
  pcont_matrix[i, pdata$cstream_inhand_left_hand_obj_all_child[i]] > 0
}))
pdata$visible_inhand_right_child = as.numeric(sapply(1:nrow(pdata), function(i) {
  pcont_matrix[i, pdata$cstream_inhand_right_hand_obj_all_child[i]] > 0
}))
pdata$visible_inhand_left_parent = as.numeric(sapply(1:nrow(pdata), function(i) {
  pcont_matrix[i, pdata$cstream_inhand_left_hand_obj_all_parent[i]] > 0
}))
pdata$visible_inhand_right_parent = as.numeric(sapply(1:nrow(pdata), function(i) {
  pcont_matrix[i, pdata$cstream_inhand_right_hand_obj_all_parent[i]] > 0
}))
pdata = pdata %>% mutate(across(starts_with('visible_inhand'), ~ifelse(is.na(.), 0, .)))
# Code the 4 holding statuses
cdata = cdata %>% mutate(visible_inhand_child = as.numeric((visible_inhand_right_child + visible_inhand_left_child) > 0),
			 visible_inhand_parent = as.numeric((visible_inhand_right_parent + visible_inhand_left_parent) > 0),
			 visible_inhand_both = as.numeric((visible_inhand_child + visible_inhand_parent) == 2),
			 visible_inhand_neither = as.numeric((visible_inhand_child + visible_inhand_parent) == 0))
pdata = pdata %>% mutate(visible_inhand_child = as.numeric((visible_inhand_right_child + visible_inhand_left_child) > 0),
                         visible_inhand_parent = as.numeric((visible_inhand_right_parent + visible_inhand_left_parent) > 0),
                         visible_inhand_both = as.numeric((visible_inhand_child + visible_inhand_parent) == 2),
                         visible_inhand_neither = as.numeric((visible_inhand_child + visible_inhand_parent) == 0))
write.csv(cdata, file.path(cfile, 'holding_status.csv'), row.names=F)
write.csv(pdata, file.path(pfile, 'holding_status.csv'), row.names = F)




