# Arguments:
args = commandArgs(trailingOnly=TRUE)
if (length(args) == 0) stop("\nArguments:
1. the path to the functions file to source
2. the path to the (unmerged) derived data folder
3. the path to the speech data file
4. path to the folder with the mindist data called 'child.csv' and 'parent.csv'
5. experiment number
6. Path where to save the resultant .csv file (with filename)
7. boolean whether to include the obj size data
8. path to the file with the obj size child data
9. path to the file with the obj size parent data")
source(args[1])
derivedpath = args[2]
speechpath = args[3]
mindistdir = args[4]
exp_num = args[5]
savefile = args[6]
if (exp_num == 15) num_obj = 10 else num_obj = 24
obj_size = args[7]
# Read in data
speech = read.csv(speechpath, stringsAsFactors = F)
c.min_dist = read.csv(file.path(mindistdir, 'child.csv'), stringsAsFactor = F)
p.min_dist= read.csv(file.path(mindistdir, 'parent.csv'), stringsAsFactors = F)
if (obj_size == TRUE) {
  c.contpath = args[8]; p.contpath = args[9]
  c.cont = read.csv(c.contpath, stringsAsFactors = F)
  p.cont = read.csv(p.contpath, stringsAsFactors = F)
  cont_nm = unique(names(c.cont), names(p.cont))
} else cont_nm = character()
# Generate the super master data frame and save it out
merge_results(derivedpath) %>% 
	make_holding_variables(num_obj = num_obj) %>% 
	make_speech_variables(speech) %>% 
	left_join(c.min_dist) %>%
	left_join(p.min_dist) %>%
	{if (obj_size) {left_join(., c.cont) %>% left_join(p.cont)} else .} %>%
	mutate(across(where(is.logical), as.numeric), no_speech = !cstream_speech_utterance) %>%
	arrange(subject, frame) %>%
	select(subject, kidID, experiment, frame, time, path, image, img, joint_attention, prediction, JA_prob, 
	       notJA_prob, not_JA, leader, target, cstream_trials, 
	       starts_with('cstream_eye'), starts_with('cstream_inhand'),
	       child_eye_parent_face, parent_eye_child_face, 
	       ends_with('_holding'), contains('speech'), contains('dist'), one_of(cont_nm)) %>%
	write.csv(savefile, row.names = F)
