% save jpg images as bin file for cpp
%
is_server = 1;

dataset = 'voc2010';  %'coco', 'voc2012'

img_folder  = './'
save_folder = './';



if ~exist(save_folder, 'dir')
    mkdir(save_folder);
end

img_dir = dir(fullfile(img_folder, '*.jpg'));

for i = 1 : numel(img_dir)
    fprintf(1, 'processing %d (%d)...\n', i, numel(img_dir));
    img = imread(fullfile(img_folder, img_dir(i).name));
    %img = imcrop(img,[50 10 320 320]);
    img_fn = img_dir(i).name(1:end-4);
    save_fn = fullfile(save_folder, [img_fn, '.ppm']);
    
    imwrite(img, save_fn);   
end
    
