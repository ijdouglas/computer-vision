library(parallel); library(dplyr); library(tidyr); library(purrr); library(readr); library(MLmetrics); library(magrittr); library(broom)
library(R.matlab); library(tibble)
merge_results = function(directory_to_derived) {
  require(parallel); require(dplyr); require(tidyr); require(purrr); require(readr)
  f = list.files(directory_to_derived, full.names=T)
  mclapply(X=f, mc.cores = 10, FUN=read.csv, stringsAsFactors = F) %>% reduce(rbind)
}
process_derived = function(path_to_derived_directory, .who, .response, balance)
{
  if (balance) {
    merge_results(path_to_derived_directory) %>%
      filter_and_balance_test(.who, .response) %>%
      make_holding_variables
  } else if (!balance) {
      merge_results(path_to_derived_directory) %>%
	extract_looks_test %>%
	make_holding_variables
  }			
}
super_confusion_df = function(merged_results) {
	require(MLmetrics)
	with(ungroup(merged_results), ConfusionDF(y_true=joint_attention, prediction))
}
super_confusion_matrix = function(merged_results) {
	require(MLmetrics)
	with(ungroup(merged_results), ConfusionMatrix(y_true=joint_attention, prediction))
}
subjectwise_Acc_AUC = function(merged_results) {
	require(MLmetrics); require(dplyr); require(tidyr)
	merged_results %>% group_by(subject) %>%
		summarise(Acc = Accuracy(y_true=joint_attention, prediction),
			  AUC = AUC(y_true=joint_attention, JA_prob))
}
super_Acc_AUC = function(merged_results) {
	require(MLmetrics); require(dplyr); require(tidyr)
	merged_results %>% ungroup %>%
		summarise(Acc = Accuracy(y_true=joint_attention, prediction),
			  AUC = AUC(y_true=joint_attention, JA_prob))
}
summary_stats = function(merged_results) {
	require(dplyr); require(tidyr)
	subjectwise_Acc_AUC(merged_results) %>%
	{t_test = t.test(.$Acc, mu = .5)
         summarise(., Mean_Accuracy = mean(Acc), SD = sd(Acc), SE = SD/sqrt(n())) %>%
		 mutate(CI.lwr = t_test$conf.int[1], CI.upr = t_test$conf.int[2], t = t_test$statistic, p = t_test$p.value)
	}
}
make_JA.target.holding_variables_child = function(merged_results) {
	merged_results %>%
		mutate(inhand_viewer_target = (target == cstream_inhand_left_hand_obj_all_child) |
           (target == cstream_inhand_right_hand_obj_all_child),
         inhand_partner_target = (target == cstream_inhand_left_hand_obj_all_parent) |
           (target == cstream_inhand_right_hand_obj_all_parent),
         # Now code the actual variables
         # ifelse(is.na(c(NA, F, T)), NA, ifelse(is.na(c(T, T, NA)), NA, c(NA, F, T) | c(T, T, NA))):
         # NA TRUE NA
         inhand_viewer.only_target = ifelse(is.na(inhand_viewer_target), NA,
                                            ifelse(is.na(inhand_partner_target), NA,
                                                   inhand_viewer_target & !inhand_partner_target)),
         inhand_partner.only_target = ifelse(is.na(inhand_viewer_target), NA,
                                             ifelse(is.na(inhand_partner_target), NA,
                                                    !inhand_viewer_target & inhand_partner_target)),
         inhand_both_target = ifelse(is.na(inhand_viewer_target), NA,
                                     ifelse(is.na(inhand_partner_target), NA,
                                            inhand_viewer_target & inhand_partner_target)),
         inhand_neither_target = ifelse(is.na(inhand_viewer_target), NA,
                                        ifelse(is.na(inhand_partner_target), NA,
                                               !inhand_viewer_target & !inhand_partner_target))) %>%
  mutate(across(starts_with('inhand'), as.numeric))
}
make_JA.target.holding_variables_parent = function(merged_results) {
	merged_results %>%
		mutate(inhand_viewer_target = (target == cstream_inhand_left_hand_obj_all_parent) |
           (target == cstream_inhand_right_hand_obj_all_parent),
         inhand_partner_target = (target == cstream_inhand_left_hand_obj_all_child) |
           (target == cstream_inhand_right_hand_obj_all_child),
         # Now code the actual variables
         # ifelse(is.na(c(NA, F, T)), NA, ifelse(is.na(c(T, T, NA)), NA, c(NA, F, T) | c(T, T, NA))):
         # NA TRUE NA
         inhand_viewer.only_target = ifelse(is.na(inhand_viewer_target), NA,
                                            ifelse(is.na(inhand_partner_target), NA,
                                                   inhand_viewer_target & !inhand_partner_target)),
         inhand_partner.only_target = ifelse(is.na(inhand_viewer_target), NA,
                                             ifelse(is.na(inhand_partner_target), NA,
                                                    !inhand_viewer_target & inhand_partner_target)),
         inhand_both_target = ifelse(is.na(inhand_viewer_target), NA,
                                     ifelse(is.na(inhand_partner_target), NA,
                                            inhand_viewer_target & inhand_partner_target)),
         inhand_neither_target = ifelse(is.na(inhand_viewer_target), NA,
                                        ifelse(is.na(inhand_partner_target), NA,
                                               !inhand_viewer_target & !inhand_partner_target))) %>%
  mutate(across(starts_with('inhand'), as.numeric))
}
JA.target.holding_status_child = function(merged_results) {
	require(MLmetrics)
	holding_vars = holding_vars = list('inhand_viewer.only_target',
					   'inhand_partner.only_target',
				 	   'inhand_both_target',
				  	   'inhand_neither_target') %>%
      	setNames(c('inhand_viewer.only_target',
			 'inhand_partner.only_target',
			 'inhand_both_target',
			 'inhand_neither_target'))
	holding = make_holding_variables_child(merged_results)
	imap_dfr(holding_vars, function(x, y) {
			 tmp = holding %>% filter(pluck(holding, x) == 1)
		       	 acc = Accuracy(y_pred = round(tmp$JA_prob), tmp$joint_attention)
		       	 cm = ConfusionMatrix(y_pred=round(tmp$JA_prob), tmp$joint_attention)
		       	 data.frame(holding_status = sub('inhand_','', sub('_target','',y)), viewer = 'parent', partner = 'child', ground_truth = 'joint_attention',
			      		   prediction_0 = cm[1,1], prediction_1 =cm[1,2], accuracy = acc, stringsAsFactors=F)
	}) 
}
JA.target.holding_status_parent = function(merged_results) {
	require(MLmetrics)
	holding_vars = holding_vars = list('inhand_viewer.only_target',
					   'inhand_partner.only_target',
				 	   'inhand_both_target',
				  	   'inhand_neither_target') %>%
      	setNames(c('inhand_viewer.only_target',
			 'inhand_partner.only_target',
			 'inhand_both_target',
			 'inhand_neither_target'))
	holding = make_holding_variables_parent(merged_results)
	imap_dfr(holding_vars, function(x, y) {
			 tmp = holding %>% filter(pluck(holding, x) == 1)
		       	 acc = Accuracy(y_pred = round(tmp$JA_prob), tmp$joint_attention)
		       	 cm = ConfusionMatrix(y_pred=round(tmp$JA_prob), tmp$joint_attention)
		       	 data.frame(holding_status = sub('inhand_','', sub('_target','',y)), viewer = 'parent', partner = 'child', ground_truth = 'joint_attention',
			      		   prediction_0 = cm[1,1], prediction_1 =cm[1,2], accuracy = acc, stringsAsFactors=F)
	}) 
}
extract_looks_test = function(merged_results) {
  merged_results %>%
    filter(cstream_eye_roi_fixation_child >0, cstream_eye_roi_fixation_parent >0)
}
filter_and_balance_test = function(merged_results, who = c('child', 'parent', 'all'), .col = 'joint_attention') {
  if (length(who) == 3) stop('who must be either `child` or `parent`')
  if (who == 'child') {
    merged_results %>% filter(cstream_eye_roi_fixation_child > 0) %>%
      group_by(subject) %>%
      mutate(nja = sum(!!rlang::ensym(.col)), nnja = sum(0^!!rlang::ensym(.col)), sampsize = min(nja, nnja)) %>% 
      group_by(subject, !!rlang::ensym(.col)) %>% 
      sample_n(size=min(sampsize)) %>% 
      ungroup
  } else if (who == 'parent') {
    merged_results %>% filter(cstream_eye_roi_fixation_parent > 0) %>%
      group_by(subject) %>%
      mutate(nja = sum(!!rlang::ensym(.col)), nnja = sum(0^!!rlang::ensym(.col)), sampsize = min(nja, nnja)) %>% 
      group_by(subject, !!rlang::ensym(.col)) %>% 
      sample_n(size=min(sampsize)) %>% 
      ungroup
  } else if (who == 'all') {
    merged_results %>% filter(cstream_eye_roi_fixation_child > 0, cstream_eye_roi_fixation_parent > 0) %>%
      group_by(subject) %>%
      mutate(nja = sum(!!rlang::ensym(.col)), nnja = sum(0^!!rlang::ensym(.col)), sampsize = min(nja, nnja)) %>% 
      group_by(subject, !!rlang::ensym(.col)) %>% 
      sample_n(size=min(sampsize)) %>% 
      ungroup 
  }
}
make_holding_variables = function(merged_results) {
  merged_results %>%
    mutate(child_holding = (cstream_inhand_left_hand_obj_all_child > 0) | (cstream_inhand_right_hand_obj_all_child > 0),
	   parent_holding = (cstream_inhand_left_hand_obj_all_parent) | (cstream_inhand_right_hand_obj_all_parent),
	   child.only_holding = ifelse(is.na(child_holding), NA, 
				       ifelse(is.na(parent_holding), NA,
					      child_holding & !parent_holding)),
	   parent.only_holding = ifelse(is.na(child_holding), NA, 
					ifelse(is.na(parent_holding), NA,
					       !child_holding & parent_holding)),
           both_holding = ifelse(is.na(child_holding), NA, 
				 ifelse(is.na(parent_holding), NA,
					child_holding & parent_holding)),
           neither_holding = ifelse(is.na(child_holding), NA, 
				    ifelse(is.na(parent_holding), NA,
					   !child_holding & !parent_holding)))
}
filter_holding_complete_cases = function(.dat) filter(.dat, complete.cases(select(.dat, ends_with('_holding'))))
# Summaries
summarize_holding_variables = function(data)
{
  all = data %>% group_by(subject) %>% summarise(`JA accuracy` = Accuracy(y_pred=round(JA_prob), joint_attention), `prop of JA` = round(sum(joint_attention)/n(),4))
  c.hold = data %>% group_by(subject) %>% summarise(`accuracy` = Accuracy(y_pred=round(JA_prob[child.only_holding == 1]), joint_attention[child.only_holding == 1]),
  `prop of holding` = round(sum(child.only_holding)/n(), 2), `prop of holding and JA` = round(sum((child.only_holding + joint_attention)==2)/n(), 4))
  p.hold = data %>% group_by(subject) %>% summarise(`accuracy` = Accuracy(y_pred=round(JA_prob[parent.only_holding == 1]), joint_attention[parent.only_holding == 1]),
  `prop of holding` = round(sum(parent.only_holding)/n(), 2), `prop of holding and JA` = round(sum((parent.only_holding + joint_attention)==2)/n(), 4))
  b.hold = data %>% group_by(subject) %>% summarise(`accuracy` = Accuracy(y_pred=round(JA_prob[both_holding == 1]), joint_attention[both_holding == 1]),
  `prop of holding` = round(sum(both_holding)/n(), 2), `prop of holding and JA` = round(sum((both_holding + joint_attention)==2)/n(), 4))
  n.hold = data %>% group_by(subject) %>% summarise(`accuracy` = Accuracy(y_pred=round(JA_prob[neither_holding == 1]), joint_attention[neither_holding == 1]),
  `prop of holding` = round(sum(neither_holding)/n(), 2), `prop of holding and JA` = round(sum((neither_holding + joint_attention)==2)/n(), 4))
  list(all=all, child.holding = c.hold, parent.holding = p.hold, both.holding = b.hold, neither.holding = n.hold)
}
# Get variables from multiwork
get_multiwork_variable = function(kidID, experiment, variable) {
  require(R.matlab)
  print('Usage: get_multiwork_variable("__20180508_20241", "experiment_15", "cstream_speech_utterance.mat")')
  var = readMat(paste0('/data/drives/multiwork/', experiment, '/included/', kidID, '/derived/', variable))[[1]][[2]]
  return(var)
}
get_speech_variables_multiwork = function(kidID, experiment)
{
  if (experiment == 'experiment_12') nam.nm = 'cstream_speech_naming_local-id.mat' else nam.nm = 'cstream_speech_naming.mat'
  utt = try(get_multiwork_variable(kidID, experiment, 'cstream_speech_utterance.mat'), silent=T)
  nam = try(get_multiwork_variable(kidID, experiment, nam.nm), silent=T)
  if ('try-error' %in% class(utt)) utt = matrix(NA, nrow=1, ncol=2)
  if ('try-error' %in% class(nam)) nam = matrix(NA, nrow=1, ncol=2)
  colnames(utt) <- c('time', 'cstream_speech_utterance')
  colnames(nam) <- c('time', 'cstream_speech_naming')
  full_join(as_tibble(utt), as_tibble(nam), all=T) %>% mutate(kidID = kidID) %>%
    filter(!is.na(time)) # occurs when one but not other speech is missing
}
# Example
#c15 = process_derived('experiment_15/derived/pct20/child/all_frames_test/', 'all', 'joint_attention') %>% 
#        filter_holding_complete_cases
#cview.15 = summarize_holding_variables(c15)
#> pview.15 = summarize_holding_variables(p15)
#> cview.12.acuity = summarize_holding_variables(c12.acuity)
#> pview.12.acuity = summarize_holding_variables(p12.acuity)
#> iwalk(cview.15, ~{write.csv(.x, paste0("experiment_15/summaries/holding_status/child-view_", .y, ".csv"))})
#> iwalk(pview.15, ~{write.csv(.x, paste0("experiment_15/summaries/holding_status/parent-view_", .y, ".csv"))})
#> iwalk(cview.12.acuity, ~{write.csv(.x, paste0("experiment_12/summaries/holding_status/child-view-with-acuity_", .y, ".csv"))})
#> iwalk(pview.12.acuity, ~{write.csv(.x, paste0("experiment_12/summaries/holding_status/parent-view-with-acuity_", .y, ".csv"))})
print('Defined functions:')
purrr::walk(names(.GlobalEnv), ~{if (is.function(get(.x))) print(.x)})
