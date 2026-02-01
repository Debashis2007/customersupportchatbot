# TechCorp Troubleshooting Guide

## Connection Issues

### Unable to Connect to TechCloud Instance

**Symptoms:**
- SSH connection timeout
- "Connection refused" error
- Cannot ping instance

**Solutions:**

1. **Check instance status**
   - Go to Dashboard > Instances
   - Verify status is "Running"
   - If stopped, click "Start"

2. **Verify security groups**
   - Navigate to Networking > Security Groups
   - Ensure SSH (port 22) is allowed
   - Check source IP restrictions

3. **Check firewall rules**
   ```bash
   # On the instance
   sudo ufw status
   sudo ufw allow 22/tcp
   ```

4. **Verify SSH key**
   - Ensure you're using the correct private key
   - Check key permissions: `chmod 600 your-key.pem`

5. **Network connectivity**
   - Verify VPC configuration
   - Check internet gateway attachment
   - Validate route tables

### API Connection Errors

**Error: "401 Unauthorized"**
- Verify API key is correct
- Check key hasn't expired
- Ensure key has required permissions

**Error: "429 Too Many Requests"**
- You've exceeded rate limits
- Implement exponential backoff
- Consider upgrading plan for higher limits

**Error: "503 Service Unavailable"**
- Check status.techcorp.com
- Retry with exponential backoff
- Use failover region if available

---

## Performance Issues

### Slow Application Response

**Diagnosis steps:**

1. **Check resource utilization**
   ```bash
   # CPU
   top -bn1 | head -20
   
   # Memory
   free -m
   
   # Disk
   df -h
   iostat -x 1 5
   ```

2. **Review application logs**
   - Check for errors or warnings
   - Look for slow query logs
   - Monitor request latency

3. **Database optimization**
   - Analyze slow queries
   - Add missing indexes
   - Consider query caching

4. **Enable caching**
   - Implement Redis/Memcached
   - Use CDN for static assets
   - Enable browser caching

### High CPU Usage

**Common causes and solutions:**

| Cause | Solution |
|-------|----------|
| Infinite loop | Review recent code changes |
| Memory leak | Restart application, fix leak |
| Inefficient queries | Optimize database queries |
| Heavy computation | Move to background jobs |
| Crypto mining malware | Security scan, reinstall |

### Memory Leaks

**Identification:**
```bash
# Monitor memory over time
watch -n 5 'free -m'

# Check process memory
ps aux --sort=-%mem | head -20
```

**Resolution:**
1. Identify leaking process
2. Review code for:
   - Unclosed connections
   - Growing data structures
   - Event listener accumulation
3. Implement proper cleanup
4. Consider automatic restarts

---

## Database Issues

### Database Connection Pool Exhausted

**Error:** "Cannot acquire connection from pool"

**Solutions:**

1. **Increase pool size**
   ```yaml
   database:
     pool:
       max: 20
       min: 5
       idle_timeout: 10000
   ```

2. **Find connection leaks**
   - Check for unclosed connections
   - Implement connection timeout
   - Use connection monitoring

3. **Optimize queries**
   - Reduce query time
   - Use connection efficiently
   - Implement query caching

### Replication Lag

**Monitoring:**
```sql
SHOW SLAVE STATUS\G
-- Check Seconds_Behind_Master
```

**Causes and solutions:**

1. **Heavy write load**
   - Batch writes
   - Optimize write queries
   - Consider sharding

2. **Network latency**
   - Use same region replicas
   - Increase network bandwidth

3. **Slave configuration**
   - Increase slave resources
   - Enable parallel replication

### Backup and Recovery

**Creating manual backup:**
```bash
# Full backup
techcloud backup create --instance my-db --type full

# Incremental backup
techcloud backup create --instance my-db --type incremental
```

**Restoring from backup:**
```bash
# List backups
techcloud backup list --instance my-db

# Restore
techcloud backup restore --backup-id bkp-123456 --target my-db-restored
```

---

## Authentication Issues

### Unable to Login

**Checklist:**

- [ ] Correct email address
- [ ] Correct password (check Caps Lock)
- [ ] Account not locked (5 failed attempts)
- [ ] Account is active
- [ ] No IP restrictions blocking access
- [ ] Browser cookies enabled
- [ ] Try incognito/private mode

**Account locked?**
1. Wait 30 minutes for auto-unlock
2. Or contact support for immediate unlock
3. Reset password after unlocking

### SSO Configuration Issues

**SAML Errors:**

| Error | Solution |
|-------|----------|
| Invalid signature | Check certificate configuration |
| Clock skew | Sync server time with NTP |
| Audience mismatch | Verify Entity ID settings |
| Missing attributes | Map required SAML attributes |

**OAuth Errors:**

| Error | Solution |
|-------|----------|
| Invalid redirect | Whitelist redirect URI |
| Invalid client | Check client ID/secret |
| Scope error | Request correct scopes |

### MFA Issues

**Lost MFA device:**
1. Use backup codes (saved during setup)
2. Contact support with ID verification
3. Admin can temporarily disable MFA

**MFA not working:**
1. Sync device time (must be accurate)
2. Try backup codes
3. Check for clock drift on server

---

## Email Delivery Issues

### Emails Not Sending

**Checklist:**

1. **Verify quota**
   - Check Settings > Email > Quota
   - Request increase if needed

2. **Check recipient**
   - Valid email format
   - Not on bounce list
   - Not on suppression list

3. **Review content**
   - No spam trigger words
   - Valid FROM address
   - Proper authentication

### Emails Going to Spam

**Solutions:**

1. **DNS Configuration**
   - Add SPF record
   - Configure DKIM
   - Set up DMARC

2. **Content optimization**
   - Avoid spam trigger words
   - Balance text and images
   - Include unsubscribe link

3. **Sender reputation**
   - Warm up new IPs gradually
   - Monitor bounce rates
   - Handle complaints promptly

### Bounce Handling

**Hard bounces (permanent):**
- Remove from mailing list
- Do not retry
- Update CRM

**Soft bounces (temporary):**
- Retry up to 3 times
- Escalate to hard bounce if persists
- Check mailbox full, server down

---

## Integration Issues

### Webhook Delivery Failures

**Debugging:**

1. **Check webhook logs**
   - Go to Settings > Webhooks > Logs
   - Review failure reasons
   - Check response codes

2. **Verify endpoint**
   - URL is accessible
   - Returns 2xx status
   - Responds within timeout (30s)

3. **Check payload**
   - Valid JSON format
   - Expected fields present
   - Size within limits

### API Rate Limiting

**Rate limits by plan:**
| Plan | Requests/minute |
|------|-----------------|
| Basic | 60 |
| Pro | 300 |
| Enterprise | Custom |

**Best practices:**
1. Implement exponential backoff
2. Cache responses when possible
3. Use batch endpoints
4. Monitor usage with headers:
   - `X-RateLimit-Remaining`
   - `X-RateLimit-Reset`

---

## Common Error Codes

| Code | Meaning | Solution |
|------|---------|----------|
| E001 | Invalid credentials | Check username/password |
| E002 | Account suspended | Contact billing |
| E003 | Resource not found | Verify resource ID |
| E004 | Permission denied | Check user permissions |
| E005 | Quota exceeded | Upgrade plan or clean up |
| E006 | Rate limited | Implement backoff |
| E007 | Invalid input | Check request format |
| E008 | Service unavailable | Check status page |
| E009 | Timeout | Retry with longer timeout |
| E010 | Maintenance mode | Wait for completion |

## Getting Help

If you can't resolve your issue:

1. **Search knowledge base**: help.techcorp.com
2. **Community forum**: community.techcorp.com
3. **Submit ticket**: support.techcorp.com/tickets
4. **Emergency support**: 1-800-TECH-911

When contacting support, include:
- Account email
- Error message/code
- Steps to reproduce
- Screenshots if applicable
- Timestamp of issue
