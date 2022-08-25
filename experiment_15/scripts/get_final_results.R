# Arguments:
# 1. the path to the functions file to source
# 2. the path to the (unmerged) derived data folder
# 3. the path to the speech data
# 4. process child or parent? (ex: 'child')
# 5. The experiment number (ex: 91)
# 6. Path (with filename) where to save the resultant .RDS file
args = commandArgs(trailingOnly=TRUE)
if (length(args) == 0) stop("\nArguments:
1. the path to the functions file to source
2. the path to the (unmerged) derived data folder
3. the path to the speech data
4. process child or parent? (ex: 'child')
5. The experiment number (ex: 91)
6. Path (with filename) where to save the resultant .RDS file")
source(args[1])
derivedpath = args[2]
speechpath = args[3]
c_or_p = args[4]
exp_num = args[5]
out = args[6]
x = final_results(derivedpath, c_or_p, response = 'joint_attention', speechpath, exp_num)
saveRDS(x, out)
