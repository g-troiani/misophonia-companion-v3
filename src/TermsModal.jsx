// File: src/TermsModal.jsx
import React, { useState } from 'react';

export default function TermsModal({ onAccept }) {
  const [checked, setChecked] = useState(false);
  return (
    <div style={{ position:'fixed', top:0, left:0, right:0, bottom:0, backgroundColor:'rgba(0,0,0,0.7)', zIndex:1000, display:'flex', alignItems:'center', justifyContent:'center', border:'4px dashed red' }}>
      <div style={{ background:'#fff', padding:'2rem', maxWidth:'600px', width:'80%', boxSizing:'border-box', maxHeight:'80vh', overflowY:'auto', borderRadius:'8px', display:'flex', flexDirection:'column', alignItems:'center', border:'4px solid blue' }}>
        <h2>Terms and Conditions</h2>
        <div style={{ overflowY:'auto', maxHeight:'50vh', textAlign:'left', width:'100%' }}>
          <p><strong>Effective Date:</strong> [Insert Date]</p>
          <p><strong>1. Acceptance of Terms</strong> By accessing or using the Misophonia Companion application ('App'), you agree to be bound by these Terms and Conditions ('Terms'). If you do not agree to these Terms, do not use the App.</p>
          <p><strong>2. Nature of the App</strong> Misophonia Companion is a digital wellness and research guide. It is intended for informational, educational, and personal support purposes only. The App does not provide medical advice, diagnosis, or treatment. It is not a licensed healthcare service and is not intended to replace professional consultation.</p>
          <p><strong>3. Eligibility</strong> You must be at least 16 years old to use the App. If you are under 18, you confirm that you have received parental or guardian consent.</p>
          <p><strong>4. Privacy and Data</strong> We respect your privacy. Our Privacy Policy explains how we collect, use, and protect your information. By using the App, you agree to the terms of our Privacy Policy.</p>
          <p>No health information (as defined under HIPAA) is collected without explicit consent.</p>
          <p>Conversations may be stored anonymously for product improvement unless you opt out.</p>
          <p>We do not sell or share your data with third parties for advertising purposes.</p>
          <p><strong>5. User Conduct</strong> You agree to use the App responsibly and not to misuse the services, including but not limited to:</p>
          <ul>
            <li>Attempting to reverse-engineer, copy, or modify the App</li>
            <li>Submitting harmful, abusive, or misleading content</li>
            <li>Interfering with the operation or integrity of the App</li>
          </ul>
          <p><strong>6. Limitation of Liability</strong> To the fullest extent permitted by law, Misophonia Companion and its creators are not liable for any direct, indirect, incidental, or consequential damages resulting from the use—or inability to use—the App. This includes, but is not limited to, psychological distress, loss of data, or misinterpretation of information.</p>
          <p><strong>7. Modifications to the Terms</strong> We reserve the right to modify these Terms at any time. Continued use of the App after changes means you accept the new Terms.</p>
          <p><strong>8. Termination</strong> We may suspend or terminate access to the App at our discretion, without notice, for conduct that violates these Terms.</p>
          <p><strong>9. Governing Law</strong> These Terms are governed by the laws of the state of [Your State], without regard to conflict of laws principles.</p>
          <p><strong>10. Contact</strong> For questions or concerns about these Terms, please email: [Your Contact Email]</p>
        </div>
        <h2>Privacy Policy</h2>
        <div style={{ overflowY:'auto', maxHeight:'30vh', textAlign:'left', width:'100%' }}>
          <p><strong>Effective Date:</strong> [Insert Date]</p>
          <p><strong>1. Introduction</strong> This Privacy Policy describes how Misophonia Companion ('we', 'us', 'our') collects, uses, and protects your personal information when you use our web and mobile application ('App').</p>
          <p><strong>2. Information We Collect</strong></p>
          <p><strong>•</strong> <em>User-Provided Information</em>: When you create an account or interact with the App, you may provide personal information such as your email address and preferences.</p>
          <p><strong>•</strong> <em>Anonymous Usage Data</em>: We may collect anonymized data on app usage, interactions, and conversation content to improve the experience.</p>
          <p><strong>•</strong> <em>Optional Personal Logs</em>: Users may opt in to mood tracking, journaling, or trigger tagging. This data is stored securely and only accessible to the user unless explicitly shared.</p>
          <p><strong>3. How We Use Your Information</strong> To provide and improve our services; to personalize your user experience; for anonymized research and development; to comply with legal obligations if required.</p>
          <p><strong>4. Data Storage and Security</strong> We use secure, encrypted servers and industry-standard protocols to store your data. Sensitive data is never shared with third parties without consent. You may request deletion of your account and data at any time.</p>
          <p><strong>5. Sharing and Disclosure</strong> We do not sell or rent your personal information. We may share anonymized data with research collaborators or analytics partners. We may disclose information if legally compelled (e.g., court order).</p>
          <p><strong>6. Your Rights and Choices</strong> You may update or delete your personal information from your profile. You may opt out of data collection for research purposes. You may contact us for a copy of any personal data we’ve stored.</p>
          <p><strong>7. Children’s Privacy</strong> Our App is not intended for children under 16. We do not knowingly collect data from individuals under this age without parental consent.</p>
          <p><strong>8. Changes to This Policy</strong> We may revise this policy from time to time. Users will be notified of significant changes. Continued use of the App indicates acceptance of the updated policy.</p>
          <p><strong>9. Contact Us</strong> If you have questions or requests related to this Privacy Policy, please contact us at: [Your Contact Email]</p>
        </div>
        <label style={{ display:'block', margin:'1rem 0', textAlign:'center', width:'100%' }}>
          <input type='checkbox' checked={checked} onChange={e => setChecked(e.target.checked)} /> I have read and agree to the Terms and Conditions and Privacy Policy
        </label>
        <button disabled={!checked} onClick={() => { localStorage.setItem('termsAccepted','true'); onAccept(); }} style={{ marginTop:'1.5rem', padding:'0.5rem 1rem' }}>Continue</button>
      </div>
    </div>
  );
}
