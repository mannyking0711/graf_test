# ENERMAN


## setup environment (using conda)

**create conda environment**

```
$ conda create --name enerman python=3.10
```

**remove conda environment**

```
$ conda env remove --name enerman
```

**activate conda environment**

```
$ conda activate enerman
```

**export conda environment**

```
(enerman) $ conda env export > conda_env_enerman.yml
```

**recreate conda environment**

```
(base) $ conda env create -f conda_env_enerman.yml
```

