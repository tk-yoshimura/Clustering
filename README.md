# Clustering
 Clustering - svm, kmeans
 
![sample](https://github.com/tk-yoshimura/Clustering/blob/main/figures/gaussiansvm_random_cost100_svm.png)  

## Requirement
 .NET 8.0
 
## Install
[Download DLL](https://github.com/tk-yoshimura/Clustering/releases)  
[Download Nuget package](https://www.nuget.org/packages/tyoshimura.clustering/)

## Usage
```csharp
GaussianSupportVectorMachine svm = new(cost: 1000, sigma: 1);

svm.Learn(positive_vectors, negative_vectors);
int posneg = svm.Classify(vector, threshold: 0.1);
```

## Licence
[MIT](https://github.com/tk-yoshimura/Clustering/blob/main/LICENSE)

## Author

[tk-yoshimura](https://github.com/tk-yoshimura)
