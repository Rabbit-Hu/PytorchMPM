ffmpeg -framerate 5 -pattern_type glob -i 'video_all/*.png' -vcodec mpeg4 -q:v 1 -filter:v fps=30 video_all.mp4
ffmpeg -framerate 5 -pattern_type glob -i 'video_pred/*.png' -vcodec mpeg4 -q:v 1 -filter:v fps=30 video_pred.mp4
ffmpeg -framerate 5 -pattern_type glob -i 'video_guess/*.png' -vcodec mpeg4 -q:v 1 -filter:v fps=30 video_guess.mp4
ffmpeg -framerate 5 -pattern_type glob -i 'video_gt/*.png' -vcodec mpeg4 -q:v 1 -filter:v fps=30 video_gt.mp4
ffmpeg -i video_all.mp4 -i video_guess.mp4 -i video_gt.mp4 -i video_pred.mp4 -filter_complex hstack=inputs=4 -vcodec mpeg4 -q:v 1 video_hstack.mp4