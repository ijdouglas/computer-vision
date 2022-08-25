library(parallel); library(dplyr); library(tidyr); library(purrr); library(readr); library(MLmetrics); library(magrittr); library(broom)
library(R.matlab); library(tibble)

framenum2_time = function(frame_num_vec, FPS = 30, offset = 30) 
{
  ((frame_num_vec / FPS) + offset) - (1/FPS)
}

merge_results = function(directory_to_derived) {
  require(parallel); require(dplyr); require(tidyr); require(purrr); require(readr)
  f = list.files(directory_to_derived, full.names=T)
  mclapply(X=f, mc.cores = 10, FUN=read.csv, stringsAsFactors = F) %>% reduce(rbind)
}
process_derived = function(path_to_derived_directory, .who, .response, balance, num_obj)
{
  if (balance) {
    merge_results(path_to_derived_directory) %>%
      filter_and_balance_test(.who, 'joint_attention') %>%
      make_holding_variables(num_obj=num_obj)
  } else if (!balance) {
      merge_results(path_to_derived_directory) %>%
	extract_looks_test %>%
	make_holding_variables(num_obj=num_obj)
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
extract_looks_test = function(merged_results, who = c('child', 'parent', 'all'), exp_num) {
  if (length(who) == 3) stop('who must be either "child" "parent" or "all"')
  if (exp_num == 15) n_obj = 10 else n_obj = 24
  if (who == 'child') {
    merged_results %>% filter(cstream_eye_roi_fixation_child %in% 1:n_obj)
  } else if (who == 'parent') {
    merged_results %>% filter(cstream_eye_roi_fixation_parent %in% 1:n_obj)
  } else merged_results %>% filter(cstream_eye_roi_fixation_child %in% 1:n_obj,
				   cstream_eye_roi_fixation_parent %in% 1:n_obj)
}
filter_and_balance_test = function(merged_results, who = c('child', 'parent', 'all'), .col = 'joint_attention', exp_num) {
  if (length(who) == 3) stop('who must be either `child` or `parent`')
  if (exp_num == 15) n_obj = 10 else n_obj = 24
  if (who == 'child') {
    merged_results %>% filter(cstream_eye_roi_fixation_child %in% 1:n_obj) %>%
      group_by(subject) %>%
      mutate(nja = sum(!!rlang::ensym(.col)), nnja = sum(0^!!rlang::ensym(.col)), sampsize = min(nja, nnja)) %>% 
      group_by(subject, !!rlang::ensym(.col)) %>% 
      sample_n(size=min(sampsize)) %>% 
      ungroup
  } else if (who == 'parent') {
    merged_results %>% filter(cstream_eye_roi_fixation_parent %in% 1:n_obj) %>%
      group_by(subject) %>%
      mutate(nja = sum(!!rlang::ensym(.col)), nnja = sum(0^!!rlang::ensym(.col)), sampsize = min(nja, nnja)) %>% 
      group_by(subject, !!rlang::ensym(.col)) %>% 
      sample_n(size=min(sampsize)) %>% 
      ungroup
  } else if (who == 'all') {
    merged_results %>% filter(cstream_eye_roi_fixation_child %in% 1:n_obj, 
			      cstream_eye_roi_fixation_parent %in% 1:n_obj) %>%
      group_by(subject) %>%
      mutate(nja = sum(!!rlang::ensym(.col)), nnja = sum(0^!!rlang::ensym(.col)), sampsize = min(nja, nnja)) %>% 
      group_by(subject, !!rlang::ensym(.col)) %>% 
      sample_n(size=min(sampsize)) %>% 
      ungroup 
  }
}
balance_test = function(merged_results, who, .col) {
	if (who == 'all') {
		merged_results %>% group_by(subject) %>%
			mutate(nja = sum(!!rlang::ensym(.col)), nnja = sum(0^!!rlang::ensym(.col)), sampsize = min(nja, nnja)) %>%
			group_by(subject, !!rlang::ensym(.col)) %>%
			sample_n(size=min(sampsize)) %>%
			ungroup
	}
}
make_holding_variables = function(merged_results, num_obj) {
  merged_results %>%
    mutate(child_holding = (cstream_inhand_left_hand_obj_all_child %in% 1:num_obj) | (cstream_inhand_right_hand_obj_all_child %in% 1:num_obj),
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
  `prop of holding` = round(sum(child.only_holding)/n(), 4), `prop of holding and JA` = round(sum((child.only_holding + joint_attention)==2)/n(), 4))
  p.hold = data %>% group_by(subject) %>% summarise(`accuracy` = Accuracy(y_pred=round(JA_prob[parent.only_holding == 1]), joint_attention[parent.only_holding == 1]),
  `prop of holding` = round(sum(parent.only_holding)/n(), 4), `prop of holding and JA` = round(sum((parent.only_holding + joint_attention)==2)/n(), 4))
  b.hold = data %>% group_by(subject) %>% summarise(`accuracy` = Accuracy(y_pred=round(JA_prob[both_holding == 1]), joint_attention[both_holding == 1]),
  `prop of holding` = round(sum(both_holding)/n(), 4), `prop of holding and JA` = round(sum((both_holding + joint_attention)==2)/n(), 4))
  n.hold = data %>% group_by(subject) %>% summarise(`accuracy` = Accuracy(y_pred=round(JA_prob[neither_holding == 1]), joint_attention[neither_holding == 1]),
  `prop of holding` = round(sum(neither_holding)/n(), 4), `prop of holding and JA` = round(sum((neither_holding + joint_attention)==2)/n(), 4))
  list(all=all, child.holding = c.hold, parent.holding = p.hold, both.holding = b.hold, neither.holding = n.hold)
}

summarize_speech_variables = function(.data.)
{
  data = .data. %>% mutate(across(starts_with('cstream_speech_'), ~as.numeric(. > 0)),
			   no_speech = as.numeric((cstream_speech_utterance + cstream_speech_naming) == 0))
  all = data %>% group_by(subject) %>%
    summarise(`JA accuracy`=Accuracy(y_pred= round(JA_prob), joint_attention), 
	      `prop of JA`= round(sum(joint_attention)/n(),4))
  utterance = data %>% group_by(subject) %>%
    summarise(accuracy= Accuracy(y_pred= round(JA_prob[cstream_speech_utterance==1]), joint_attention[cstream_speech_utterance==1]),
	      `prop of speech` = round(sum(cstream_speech_utterance)/n(), 4), 
	      `prop of speech and JA` = round(sum((cstream_speech_utterance + joint_attention) == 2)/n(), 4))
  naming = data %>% group_by(subject) %>%
	  summarise(accuracy= Accuracy(y_pred= round(JA_prob[cstream_speech_naming==1]), joint_attention[cstream_speech_naming==1]),
		    `prop of speech`= round(sum(cstream_speech_naming)/n(), 4),
		    `prop of speech and JA`= round(sum((cstream_speech_naming + joint_attention) == 2)/n(), 4))
  nospeech = data %>% group_by(subject) %>%
	  summarise(accuracy= Accuracy(y_pred= round(JA_prob[no_speech==1]), joint_attention[no_speech==1]),
		    `prop of speech`= round(sum(no_speech)/n(), 4),
		    `prop of speech and JA`= round(sum((no_speech + joint_attention) == 2)/n(), 4))
  list(all=all, utterance = utterance, naming = naming, "no-speech"=nospeech)
}
# summarization functions that use different filtering based on the output measure
holding_summary = function(merged_results, child_or_parent = c('all', 'child', 'parent'), 
			   response = 'joint_attention', exp_num)
{
  require(rlang)
  if (length(child_or_parent) == 3) stop('Supply either "all", "child" or "parent" to arg "child_or_parent"')
  if (exp_num == 15) num_obj = 10 else num_obj = 24
  #d = merge_results(path_to_derived)
  d = merged_results
  d.balanced = d %>% filter_and_balance_test(who=child_or_parent, .col=!!ensym(response), exp_num=exp_num) %>% 
    make_holding_variables(num_obj = num_obj) %>% filter_holding_complete_cases
  d.fltr = d %>% extract_looks_test(who=child_or_parent, exp_num=exp_num) %>% 
    make_holding_variables(num_obj=num_obj) %>% filter_holding_complete_cases
  accuracy.list = list(
    child.holding = d.balanced %>% filter(child.only_holding == 1) %>% group_by(subject) %>% 
      summarise(accuracy = Accuracy(y_pred=round(JA_prob), joint_attention)),
    parent.holding = d.balanced %>% filter(parent.only_holding == 1) %>% group_by(subject) %>% 
      summarise(accuracy=Accuracy(y_pred=round(JA_prob), joint_attention)),
    both.holding = d.balanced %>% filter(both_holding == 1) %>% group_by(subject) %>% 
      summarise(accuracy=Accuracy(y_pred=round(JA_prob), joint_attention)),
    neither.holding = d.balanced %>% filter(neither_holding == 1) %>% group_by(subject) %>% 
      summarise(accuracy=Accuracy(y_pred=round(JA_prob), joint_attention))
  )
  proportions.list = list(
    child.holding =d.fltr %>% 
      group_by(subject) %>% 
      summarise(`prop of holding` = round(sum(child.only_holding)/n(), 4), 
		`prop of holding and JA` = round(sum((child.only_holding + joint_attention)==2)/n(), 4)),
    parent.holding = d.fltr %>%
      group_by(subject) %>%
      summarise(`prop of holding` = round(sum(parent.only_holding)/n(), 4), 
		`prop of holding and JA` = round(sum((parent.only_holding + joint_attention)==2)/n(), 4)),
    both.holding = d.fltr %>%
      group_by(subject) %>%
      summarise(`prop of holding` = round(sum(both_holding)/n(), 4), 
		`prop of holding and JA` = round(sum((both_holding + joint_attention)==2)/n(), 4)),
    neither.holding = d.fltr %>%
      group_by(subject) %>%
      summarise(`prop of holding` = round(sum(neither_holding)/n(), 4), 
		`prop of holding and JA` = round(sum((neither_holding + joint_attention)==2)/n(), 4))
  )
  out = lapply(names(accuracy.list), function(x) {
    merge(accuracy.list[[x]], proportions.list[[x]])
  })

  # confusion matrices
  vars = c('child.only_holding', 'parent.only_holding', 'both_holding', 'neither_holding')
  conf = lapply(vars, function(x) {
    d.balanced %>% filter_at(x, ~.==1) %>% super_confusion_matrix
  })
  out = lapply(names(accuracy.list), function(x) {
    merge(accuracy.list[[x]], proportions.list[[x]])
  })
  # Return
  names(conf) <- names(out) <- names(accuracy.list)
  return(list('confusion_matrices' = conf, subjectwise_results = out))
}
#speech_summary
speech_summary = function(merged_results, child_or_parent = 'all', 
			  response = 'joint_attention', path_to_speech, exp_num)
{
  require(rlang)
  speech = read.csv(path_to_speech, stringsAsFactors = F)
  #d = merge_results(path_to_derived)
  d = merged_results
  d.balanced = d %>% 
    make_speech_variables(., speech) %>%
    filter_and_balance_test(who = child_or_parent, .col = !!ensym(response), exp_num = exp_num)
  d.fltr = d %>%
    make_speech_variables(., speech) %>%
    extract_looks_test(who = child_or_parent, exp_num)
  # Compute stats
  accuracy.list = list(
    utterance = d.balanced %>% filter(cstream_speech_utterance == 1) %>% group_by(subject) %>% 
      summarise(accuracy=Accuracy(y_pred=round(JA_prob), joint_attention)),
    naming=d.balanced %>% filter(cstream_speech_naming== 1) %>% group_by(subject) %>% 
      summarise(accuracy=Accuracy(y_pred=round(JA_prob), joint_attention)),
    nospeech=d.balanced %>% filter(no_speech== 1) %>% group_by(subject) %>% 
      summarise(accuracy=Accuracy(y_pred=round(JA_prob), joint_attention))
  )
  proportions.list = list(
    utterance=d.fltr %>% group_by(subject) %>%
      summarise(`prop of speech`= round(sum(cstream_speech_utterance)/n(), 4),
		`prop of speech and JA`= round(sum((cstream_speech_utterance+joint_attention) == 2)/n(), 4)),
    naming=d.fltr %>% group_by(subject) %>%
      summarise(`prop of speech`= round(sum(cstream_speech_naming)/n(), 4),
		`prop of speech and JA`= round(sum((cstream_speech_naming+joint_attention) == 2)/n(), 4)),
    nospeech=d.fltr %>% group_by(subject) %>%
      summarise(`prop of speech` = round(sum(no_speech)/n(), 4),
                `prop of speech and JA`=round(sum((no_speech + joint_attention) == 2)/n(), 4))
  )
  # confusion matrices
  vars = c('cstream_speech_utterance', 'cstream_speech_naming', 'no_speech')
  conf = lapply(vars, function(x) {
    d.balanced %>% filter_at(x, ~.==1) %>% super_confusion_matrix
  })
  out = lapply(names(accuracy.list), function(x) {
    merge(accuracy.list[[x]], proportions.list[[x]])
  })
  # Return
  names(conf) <- names(out) <- names(accuracy.list)
  return(list('confusion_matrices' = conf, subjectwise_results = out))
}
# Helper function for whole pop. accuracy:
super_Accuracy = function(merged_results) {
	require(MLmetrics); require(dplyr); require(tidyr)
	merged_results %>% ungroup %>%
		summarise(Acc = Accuracy(y_true=joint_attention, prediction))
}
# Final results function:
final_results = function(derived_path, who = 'all', response = 'joint_attention', speech_path, exp_num) 
{
  d = merge_results(derived_path)
  population.prop.JA = d %>% summarise(`prop of JA` = sum(joint_attention > 0, na.rm=T) / n())
  subjectwise.prop.JA = d %>% group_by(subject) %>% summarise(`prop of JA`=sum(joint_attention>0,na.rm=T) / n())
  holding_results = holding_summary(d, who, !!ensym(response), exp_num)
  speech_results = speech_summary(d, who, !!ensym(response), speech_path, exp_num)
  subjectwise_results = d %>% filter_and_balance_test(who, !!ensym(response), exp_num) %>% subjectwise_Acc_AUC
  super_Acc = d %>% filter_and_balance_test(who, !!ensym(response), exp_num) %>% super_Accuracy
  super_conf = d %>% filter_and_balance_test(who, !!ensym(response), exp_num) %>% super_confusion_matrix
  list(full_sample = list(accuracy = super_Acc, confusion.matrix=super_conf, 
			  population.prop.JA = population.prop.JA, subjectwise.prop.JA = subjectwise.prop.JA),
       by_categories = list(subjectwise.accuracy = subjectwise_results, holding = holding_results, speech = speech_results))  
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
  #if (experiment == 'experiment_12') nam.nm = 'cstream_speech_naming_local-id.mat' else nam.nm = 'cstream_speech_naming.mat'
  nam.nm = 'cstream_speech_naming_local-id.mat'
  utt = try(get_multiwork_variable(kidID, experiment, 'cstream_speech_utterance.mat'), silent=T)
  nam = try(get_multiwork_variable(kidID, experiment, nam.nm), silent=T)
  if ('try-error' %in% class(utt)) utt = matrix(NA, nrow=1, ncol=2)
  if ('try-error' %in% class(nam)) nam = matrix(NA, nrow=1, ncol=2)
  colnames(utt) <- c('time', 'cstream_speech_utterance')
  colnames(nam) <- c('time', 'cstream_speech_naming')
  full_join(as_tibble(utt), as_tibble(nam), all=T) %>% mutate(kidID = kidID) %>%
    filter(!is.na(time)) %>% # occurs when one but not other speech is missing
    mutate(frame = time2frame_num(time)) %>%
    filter(!is.na(cstream_speech_utterance), !is.na(cstream_speech_naming))
}
get_min.dist_data.frame = function(path, n_obj)
{
  print('Note: path is the path to the data extracted from matlab with time as first column cont_vision_min-dist_obj%d-to-center_child as all other columns')
  require(stringr)
  if (grepl('child', path)) who = 'child' else who = 'parent'
  f = list.files(path, full.names=T)
  SUBJECTS = str_extract(f, '[0-9]{4}')
  d = lapply(f, read.csv, stringsAsFactors = F, header = F) %>%
    map(~{setNames(.x, c('time', paste0('dist_obj', 1:n_obj, '_to_center_', who)))}) %>%
    setNames(SUBJECTS) %>%
    imap(~{mutate(.x, across(everything(), as.numeric), frame = time2frame_num(time), subject = .y)}) %>%
    reduce(rbind) %>%
    select(subject, time, frame, everything())
}
time2frame_num = function(time, sampling_rate = 30L, offset = 30.0)
{
  round(sampling_rate * (time - offset) + 1)
}
make_speech_variables = function(processed_derived, speech_data) {
  # Process derived should be outputs of `process_derived`
  # Example of a speech_data file: 
  # 'experiment_12/data/speech/child-and-parent_cstream-speech-utterance_cstream-speech-naming.csv'
  left_join(processed_derived, speech_data) %>%
    mutate(across(starts_with('cstream_speech_'), ~as.numeric(. > 0)),
	   no_speech = as.numeric((cstream_speech_utterance + cstream_speech_naming) == 0))
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

# For speech:
# c.processed_derived.12.binary = process_derived('experiment_12/derived/child/pct40/binary', 'all', 'joint_attention', F)
# speech12 = read.csv('experiment_12/data/speech/child-and-parent_cstream-speech-utterance_cstream-speech-naming.csv', stringsAsFactors = F)
# c.processed_speech.12.binary = left_join(c.processed_derived.12.binary, speech12)
# c.speech_summaries.12.binary = summarize_speech_variables(c.processed_speech.12.binary)
# iwalk(c.speech_summaries.12.binary, ~{write.csv(.x, paste0("experiment_12/summaries/speech/child-view_", .y, ".csv"), row.names=F)})
print('Defined functions:')
purrr::walk(names(.GlobalEnv), ~{if (is.function(get(.x))) print(.x)})
