clear; clc; close all;

% Each line contains an ID and a list of settings for normalization
% and descriptor parameters. Here we only test default settings.
% For the meaning of each parameter, see below.
cases = [...
    %0  1    0    0  0  0  0 0  0.0  0 0    % no normalization
    1   0.2  1   -2  0  0  0 10 0.09 1 6;   % default setting
    %2  1    1   -2  0  0  0 10 0.09 1 6;   % no gamma
    %3  0.2  0    0  0  0  0 10 0.09 1 6;   % no DoG
    %4  0.2  1   -2  0  0  0 0  0.09 1 6;   % no equalization
    %5  0.2  1   -2  0  0  0 -10  0.09 1 6; % no tanh compression
    ];

for i = 1:size(cases,1)

    %*****************************************************
    c = cases(i,:);
    casenum = c(1);  % type of the parameter settings

    % parameter setting for preprocessing
    gamma = c(2);    % gamma parameter
    sigma0 = c(3);   % inner Gaussian size
    sigma1 = c(4);   % outer Gaussian size
    sx = c(5);       % x offset of centres of inner and outer filter
    sy = c(6);       % y offset of centres of inner and outer filter
    mask = c(7);     % mask
    do_norm = c(8);  % Normalize the spread of output values

    % parameter setting for LTP code and Distance Transform-based similarity
    % metric calculation
    thresh= c(9);       % threshold for LTP code
    dtalpha =c(10);     % alpha parameter for DT based distance
    dtthresh = c(11);   % threshold for truncating DT distance
    %*****************************************************

    if mask
       load('mask.mat');
       mask = double(mask1);
    else
       mask = [];
    end

    % Andrea Corriga    
    rootFolder='YaleFaces';

    %categories extraction
    dirs = dir(rootFolder);
    categories={dirs(:).name};
    categories(ismember(categories,{'.','..'})) = []; %for windows users
    imds=imageDatastore(fullfile(rootFolder,categories), 'FileExtensions', '.pgm', 'LabelSource','foldernames', 'IncludeSubfolders', true);

    images_number = length(imds.Files);
    modified_rootFolder = ['Modified' rootFolder];

    if(7 ~= exist(modified_rootFolder, 'dir'))
        mkdir(modified_rootFolder);
    end
    
   for j=1:images_number
        I = double(imread(imds.Files{j}));
        Imodified = preproc2(I,gamma,sigma0,sigma1,[sx,sy],mask,do_norm);

        Imodified=Imodified-min(Imodified(:)); % shift data such that the smallest element of A is 0
        Imodified=Imodified/max(Imodified(:)); % normalize the shifted data to 1 

        [path, image_name, ext] = fileparts(imds.Files{j});

        new_path = [modified_rootFolder '/' char(imds.Labels(j))];
        if(7 ~= exist(new_path, 'dir'))
            mkdir(new_path);
        end

        imwrite(Imodified, [new_path '/' image_name ext]);
   end

    
end % end for i : case