%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           Deskew the data by simply shifting the images laterally 
%           Currently working on the images that move up and down (not left
%           and right). If you want to process the data left and right,
%           please rotate the data first.
%           Note: 
%           1. You can choose if you want to save the deskew(shear) images.
%           2. BigTIFF format is automatically used for files larger than 4GB,
%           preserving full resolution for all output images.
%           3. Fixed some bugs about artificial strips in the topview. It
%           was discovered by Tadamoto Isogai
%           Bo-Jui, 2025/5/9 @ Dallas
%           Updated 2025/12/3: Added BigTIFF support to preserve full resolution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Configuration Parameters

% =========================================================================
% INJECTED BY NEXTFLOW / PYTHON WRAPPER
% the following variables are passed directly into the workspace:
% imagePath, CellName, CellIndex, dx, dz, angle, flip
% DO NOT hardcode them here or it will break the pipeline
% =========================================================================

% set defaults for variables that might not be passed by the python wrapper
if ~exist('ChannelsToProcess', 'var')
    ChannelsToProcess = [1]; % Specify channels to process (starting from 0)
end
if ~exist('timepoints', 'var')
    timepoints = []; % Leave blank to process all
end
if ~exist('save', 'var')
    save = 0; % save shear image? if yes, save = 1, else save = 0
end

tic;
%% Processing Setup
numFolders = numel(CellIndex);
numChannels = numel(ChannelsToProcess);

for c = 1:numFolders
    % Create folder for output
    cellNameWithIndex = strcat(CellName, num2str(CellIndex(c)));
        
    % Determine number of images
    numImages = numel(dir(fullfile(imagePath, cellNameWithIndex))) - 3;
    
    % Define timepoints range
    if isempty(timepoints)
        t_start = 0;
        t_end = round(numImages / numChannels) - 1;
    else
        t_start = min(timepoints);
        t_end = max(timepoints);
    end
    
    % Process each timepoint and channel
    for t = t_start:t_end
        for ch = 1:numChannels
            tic;
            
            % Construct filename and read image stack
            filename=strcat('1_CH',num2str(ChannelsToProcess(ch),'%02.0f'),'_',num2str((t),'%06.0f'),'.tif');
            filepath = fullfile(imagePath, cellNameWithIndex, filename);
            Image = readtiffstack(filepath);
            
            % Image dimensions
            [ysize, xsize, zsize] = size(Image);
            %pady = ysize;
            newdz=dz*cosd(angle);
            padx = xsize;
            pady = round((newdz/dx)*zsize/2);
            
            %% Apply Shearing Transformation
            disp("Deskewing ... ");
            clear ShearImage 
            %[~, Index] = max(Image(:));
            %[cx, cy, cz] = ind2sub(size(Image), Index);
            cz = floor(zsize/2)+1;
            
            max_yoffset= abs(round(flip * (1 - cz) * (newdz / dx)));
            %temp = zeros(ysize+2*max_yoffset,xsize);
            ShearImage = zeros(ysize+2*max_yoffset,xsize,zsize);
                      
            tic
            for z = 1:zsize
                yoffset = round(flip * (z - cz) * (newdz / dx)); % Ensure integer index
                %yoffset = flip * (z - cz) * (newdz / dx);
                %temp = repmat(Image(:,:,z), 4, 1);
                %temp = temp(ysize-pady+1:2*ysize+pady, xsize-padx+1:2*xsize+padx);
                %temp = temp(:, xsize-padx+1:2*xsize+padx);
                
                ShearImage(yoffset+max_yoffset+1:ysize+yoffset+max_yoffset,:,z) = Image(:,:,z);
            end
            
            
     % A little average to reduce the artifact ringing effect on the topview
                   
            for z = 1:zsize-1;
            ShearImage(:,:,z) = (ShearImage(:,:,z)+ShearImage(:,:,z+1))/2;
            end  
            
            ShearImage=uint16(ShearImage);

            toc
            % Save the processed image stack
            disp("Saving the shear image");
            tic
            output_size = size(ShearImage,1)*size(ShearImage,2)*size(ShearImage,3)*2/(1024*1024*1024); %In GB)
            if save == 1
                outputFolder = fullfile(imagePath, strcat('shear',num2str(angle),'_mlv2_',cellNameWithIndex));
                mkdir(outputFolder);
                if output_size > 4
                    disp(sprintf("File is larger than 4GB (%.2f GB), saving as BigTIFF format", output_size));
                end
                writetiffstack(ShearImage, fullfile(outputFolder, filename));
            else
                disp("Shear image saving is disabled (save=0)");
            end
            toc
            %% Rotation to top view
            disp("Rotating to top view ... ");
            clear mipzy scaled_ShearImage scaled_mipzy rot_scaled_mipzy mask cropped_mipzy zy_view cropped_rotate_zy;
            mipzy(:,:) = max(ShearImage, [], 2);
            figure(1) 
            imagesc(mipzy); axis equal tight
            
            scale_x = dz*sind(angle)/dx;  % Scaling factor in x-direction
            scale_y = 1;  % Keep y-direction the same

            % Resize only in x-direction
            scaled_ShearImage = imresize3(ShearImage, [size(ShearImage,1), size(ShearImage,2), round(size(ShearImage,3) * scale_x)],'Method','linear');
            scaled_mipzy(:,:) = max(scaled_ShearImage, [], 2);
            figure(2) 
            imagesc(scaled_mipzy); axis equal tight
            
            rot_scaled_mipzy(:,:) = imrotate(scaled_mipzy, -1*flip*angle,'bilinear'); % Rotate image
            figure(2)
            imagesc(rot_scaled_mipzy); axis equal tight
            
            % Find the bounding box of nonzero pixels
            mask = rot_scaled_mipzy > 0;
            [row, col] = find(mask); 
            min_row = min(row); max_row = max(row);
            min_col = min(col); max_col = max(col);

            % Crop the MIPyz
            cropped_mipzy = rot_scaled_mipzy(min_row:max_row, min_col:max_col);
            figure(2)
            imagesc(cropped_mipzy); axis equal tight
            
            % Crop the whole image in yz
            zy_view = permute(scaled_ShearImage, [1 ,3, 2]); % Swap dimensions to get ZY slices
            
            % Rotate each slice in the stack
            tic
            rotated_zy = single(zeros(size(rot_scaled_mipzy,1),size(rot_scaled_mipzy,2))); % Preallocate rotated volume.
            % I started to use "single" on 20251016 because the array was too
            % big (out of memory) when processing Seweryn's data

            i = 1:size(zy_view, 3);
            rotated_zy(:,:,i) = imrotate(zy_view(:,:,i), -1*flip*angle,'bilinear');
           
            %cropped_rotate_zy(:,:,i) = rotated_zy(1:1609, 443:1168,i);
            cropped_rotate_zy(:,:,:) = rotated_zy(min_row:max_row, min_col:max_col,:);
            toc
            
            rotTop_ShearImage = permute(cropped_rotate_zy, [1 3 2]); % Swap dimensions to get ZY slices
            rotTop_ShearImage = uint16(rotTop_ShearImage);
            output_size_rotTop = size(rotTop_ShearImage,1)*size(rotTop_ShearImage,2)*size(rotTop_ShearImage,3)*2/(1024*1024*1024); %In GB)
                        
            outputFolder2 = fullfile(imagePath, strcat('Top_shear',num2str(angle),'_mlv2_',cellNameWithIndex));
            mkdir(outputFolder2);
            
            disp("Saving the top-view image");
            if output_size_rotTop > 4
                disp("File is larger than 4GB, saving as BigTIFF format");
            end
            tic

            % Write full resolution image (writetiffstack handles BigTIFF automatically)
            writetiffstack(rotTop_ShearImage, fullfile(outputFolder2, filename));

            % Write note file
            msg = 'z pixel = x(y) pixel. Full resolution preserved.';
            if output_size_rotTop > 4
                msg = sprintf('%s BigTIFF format used for %.2f GB file.', msg, output_size_rotTop);
            end
            note_filename = fullfile(outputFolder2, 'note.txt');
            fileID = fopen(note_filename,'w');
            fprintf(fileID,msg);
            fclose(fileID);
            toc;
            toc; 
        end
    end
end
toc
disp("All processing completed successfully.");