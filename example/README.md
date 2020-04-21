# Example of file structrue

* Cheatsheet of `gocryptfs`
  * reverse mode means decryption
  * normal mode means encryption

  ```
  # init reverse-mode on user's home dir
  gocryptfs -init -reverse /home/user

  # mount dir contains encrypted data of /home/user
  mkdir /tmp/crypt
  gocryptfs -reverse /home/user /tmp/crypt

  # copy the encrypted data & remove the mount point
  cp -a /tmp/crypt /tmp/backup
  fusermount -u /tmp/crypt
  rmdir /tmp/crypt
   
   # restore the original(decrypted) data by normal mount
   mkdir /tmp/restore
   gocryptfs /tmp/backup/ /tmp/restore
  ```

## 2. Retinopathy data
### Plain (Decrypted)
* `tree dec_resnet`
![](https://i.imgur.com/6MwErU4.png)
TL;DR
![](https://i.imgur.com/cCT1MP1.png)
![](https://i.imgur.com/pVMLehe.png)

### Encrypted by `gocryptfs`
* `tree enc_resnet`
![](https://i.imgur.com/kPiGRyO.png)


## 1. MNIST & FashinMNIST 

### Plain (Decrypted)
![](https://i.imgur.com/RfFVTBD.png)

### Encrypted by `gocryptfs`
![](https://i.imgur.com/sGkj9Jw.png =300x)
![](https://i.imgur.com/YfpJzHu.png =350x)
