
function [nn] = nn_import_simple( path, name )
      
      str = fileread(path);
      data = jsondecode(str);
      
      dims = data.( name ).topology;
      
      nn = struct('weights', {cell(1, size(dims,1)-1 )}, ...
            'biases', {cell(1, size(dims,1)-1 )} );

      for i = 1:size(dims,1)-1
            W = reshape( [data.( name ).data(i).weight], dims(i), dims(i+1) )';
            B = reshape( [data.( name ).data(i).bias], dims(i+1), 1 );
            
            nn.weights{i} = W;
            nn.biases{i} =B;
      end
end
