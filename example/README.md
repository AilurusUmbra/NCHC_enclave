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
  
## 1. MNIST & FashinMNIST 

### Plain (Decrypted)

<img src="https://i.imgur.com/RfFVTBD.png" width=270>

### Encrypted by `gocryptfs`

<img src="https://i.imgur.com/sGkj9Jw.png" width=300>
<img src="https://i.imgur.com/YfpJzHu.png" width=350>

---

## 2. Retinopathy data
### Plain (Decrypted)
* `tree dec_resnet`
<img src="https://i.imgur.com/6MwErU4.png" width=270>
<img src="https://i.imgur.com/cCT1MP1.png" width=270>
<img src="https://i.imgur.com/pVMLehe.png" width=270>

### Encrypted by `gocryptfs`
* `tree enc_resnet`

<img aligh="right" src="https://i.imgur.com/kPiGRyO.png" width=380>

