c.fm = read.csv('../data/filtered_frames-master_child.csv', stringsAsFactors=F)
p.fm = read.csv('../data/filtered_frames-master_parent.csv', stringsAsFactors=F)
data.list = list('child' = c.fm, 'parent' = p.fm)
chronological_split = function(data)
{
  require(dplyr)
  train = data %>% slice(1:round(nrow(.)*.8)) %>% mutate(partition = 'train')
  test = data %>% slice((round(nrow(.)*.8) + 1):nrow(.)) %>% mutate(partition = 'test')
  train = train %>%
    mutate(n_JA = sum(joint_attention, na.rm=T), n_not_JA = sum(not_JA, na.rm=T)) %>%
    group_by(joint_attention) %>%
    sample_n(size = min(unique(n_JA), unique(n_not_JA)))
  test = test %>%
    mutate(n_JA = sum(joint_attention, na.rm=T), n_not_JA = sum(not_JA, na.rm=T)) %>%
    group_by(joint_attention) %>%
    sample_n(size = min(unique(n_JA), unique(n_not_JA)))
rbind(train, test)
}
data.list = data.list %>% map(~split.data.frame(.x, .x$subject))
require(parallel)
res = lapply(data.list, function(viewer.list) {mclapply(X=viewer.list, FUN=chronological_split, mc.cores = 20)})
nms = lapply(res, function(x) sapply(x, function(y) unique(y$subject)))
names(res$child) <- nms$child
names(res$parent) <- nms$parent
iwalk(res$child, ~{write.csv(.x %>% filter(partition == 'train'), paste0('../data/chronological/child/train/', .y, '_train.csv'), row.names=F)})
iwalk(res$child, ~{write.csv(.x %>% filter(partition == 'test'), paste0('../data/chronological/child/test/', .y, '_test.csv'), row.names=F)})
iwalk(res$parent, ~{write.csv(.x %>% filter(partition == 'train'), paste0('../data/chronological/parent/train/', .y, '_train.csv'), row.names=F)})
iwalk(res$parent, ~{write.csv(.x %>% filter(partition == 'test'), paste0('../data/chronological/parent/test/', .y, '_test.csv'), row.names=F)})
