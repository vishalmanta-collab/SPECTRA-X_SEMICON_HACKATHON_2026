# SPECTRA-X
## AI-Based Restoration of Degraded Semiconductor Inspection Images

SPECTRA-X is a lightweight deep-learning image restoration system developed for the **SEMICON India Hackathon 2026 – KLA Problem Statement 1**.

The system is designed to restore degraded semiconductor inspection images by suppressing noise, increasing spatial resolution, and recovering fine structural details while maintaining practical inference speed.

---

## Team

**Ashmita Sarkar**  
**Rakhi Kalowar**  
**Rabinath Goswami**  
**Vishal Manta**

**Department of Physics**  
**Indian Institute of Technology Guwahati**

---

## Problem Statement

Semiconductor inspection images can suffer from degradation such as noise, reduced spatial resolution, and loss of fine structural information.

SPECTRA-X addresses this problem using a lightweight restoration network that performs simultaneous:

- Noise suppression
- Structural restoration
- Edge preservation
- 2× super-resolution

### Input → Output

```text
Degraded Low-Resolution Image
128 × 128
        │
        ▼
    SPECTRA-X
        │
        ▼
Restored High-Resolution Image
256 × 256
