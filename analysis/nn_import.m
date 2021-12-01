
function [nn] = nn_import( path )
      
      str = fileread(path);
      data = jsondecode( str );
      
              
      b =  cell({});
      W =  cell({});

      P = [];
      lambda = cell({});
      alphas = cell({});
      betas = cell({});
      
      keys = fieldnames(data);
      
      for k = 1:length(keys)
          arr = data.( keys{k} );
         
          
          if contains( keys{k}, 'lyapu')
              shape = [ arr.cols, arr.rows ];
              P = reshape( arr.data, shape )'; 
          end
          
          if contains( keys{k}, 'weights')
              for i = 1:length(arr)
                  shape = [ arr( i ).cols, arr( i ).rows ];
                  W{ i } =  reshape( arr( i ).data, shape )';
              end 
          end
          
          if contains( keys{k}, 'biases')
              for i = 1:length(arr)
                  shape = [ arr( i ).size, 1 ];
                  b{ i } =  reshape( arr( i ).data, shape );
              end 
          end
          
          
          if contains( keys{k}, 'tvecs') || contains( keys{k}, 'lambdas')
              for i = 1:length(arr)
                  shape = [ arr( i ).size, 1 ];
                  lambda{ i } =  reshape( arr( i ).data, shape );
              end 
          end
          
          
          if contains( keys{k}, 'alphas')
              for i = 1:length(arr)
                  shape = [ arr( i ).size, 1 ];
                  alphas{ i } =  reshape( arr( i ).data, shape );
              end 
          end
          
          if contains( keys{k}, 'betas')
              for i = 1:length(arr)
                  shape = [ arr( i ).size, 1 ];
                  betas{ i } =  reshape( arr( i ).data, shape );
              end 
          end
       
      end

      
      nn = struct( 'weights', {W}, 'biases', {b}, 'lyapu', P, 'lambdas', {lambda}, 'alphas', {alphas}, 'betas', {betas});
      
      
%       nn = struct('weights', {cell(1, size(dims,1)-1 )}, ...
%             'biases', {cell(1, size(dims,1)-1 )} );
% 
%       for i = 1:size(dims,1)-1
%             W = reshape( [data.( name ).data(i).weight], dims(i), dims(i+1) )';
%             B = reshape( [data.( name ).data(i).bias], dims(i+1), 1 );
%             
%             nn.weights{i} = W;
%             nn.biases{i} =B;
%       end
end