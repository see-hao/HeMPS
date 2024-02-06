function src_models =loadSourceData()
curdir = pwd;
maindir=strcat(curdir,'\e1_model\');

subdir  = dir( maindir);

t=1;
for i = 1 : length( subdir )
    if( isequal( subdir( i ).name, '.' )||...
            isequal( subdir( i ).name, '..'))               
        continue;
    end
    
    datpath = fullfile( maindir, subdir( i ).name);
    load( datpath );
    
    src_models{1, i-2} = model;
    
    t=t+1;
    
    
end

