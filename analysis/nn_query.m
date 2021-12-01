

function y = nn_query( nn, activation,  x )


      weights = nn.weights;
      biases = nn.biases;
      
      layers = size(weights,2);
      dims = cell(1,layers);

      for i = 1:layers
            dims{i} = size(weights{i},2);
      end

      
      z = x;
      
      if isempty(biases)
          for i = 1:layers-1
                z = activation( weights{i} * z );
          end
          y =  weights{layers} * z;
      else
          for i = 1:layers-1
                z = activation( weights{i} * z + biases{i} );
          end
          y =  weights{layers} * z + biases{layers};
      end
      
end