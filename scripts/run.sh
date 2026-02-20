rsync -chavzP --stats /Users/archana/Desktop/repos/gaProj/GA_POCS  hendrix:/home/gkt175/arcn

folder_name=$(date +%Y%m%d_%H%M%S)
 
ssh hendrix "mkdir -p /home/gkt175/arcn/runs/$folder_name"

ssh hendrix "cd /home/gkt175/arcn/runs/$folder_name; sbatch --wait ../../GA_POCS/scripts/test_run.sh"

rsync -chavzP --stats hendrix:/home/gkt175/arcn/runs/$folder_name /Users/archana/Desktop/repos/gaProj/runs/$folder_name

echo "Job finished at $(date), results are in /Users/archana/Desktop/repos/gaProj/runs/$folder_name"