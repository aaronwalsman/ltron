import math
import os
import tarfile

from ltron.rollout import rollout

def generate_tar_dataset(
    name,
    total_episodes,
    shards=1,
    start_shard=0,
    save_episode_frequency=256,
    path='.',
    **kwargs,
):
    
    episodes_per_shard = math.ceil(total_episodes/shards)
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    new_shards = []
    for shard in range(shards):
        shard_name = '%s_%04i.tar'%(name, shard+start_shard)
        shard_path = os.path.expanduser(os.path.join(path, shard_name))
        new_shards.append(shard_path)
        print('Making Shard %s'%shard_path)
        shard_tar = tarfile.open(shard_path, 'w')
        shard_seqs = 0
        while shard_seqs < total_episodes:
            pass_episodes = min(
                episodes_per_shard-shard_seqs, save_episode_frequency)
            pass_name = name + ' (%i-%i/%i)'%(
                shard_seqs, shard_seqs+pass_episodes, total_episodes)
            episodes = rollout(
                pass_episodes,
                **kwargs,
            )
            print('Adding Sequences To Shard')
            save_ids = None
            if episodes.num_finished_seqs() > pass_episodes:
                save_ids = list(episodes.finished_seqs)[:pass_episodes]
            episodes.save(
                shard_tar,
                finished_only=True,
                seq_ids=save_ids,
                seq_offset=shard_seqs,
            )
            #shard_seqs += episodes.num_finished_seqs()
            shard_seqs += pass_episodes
    
    return new_shards
